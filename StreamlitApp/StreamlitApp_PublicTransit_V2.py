import streamlit as st
import pandas as pd
import googlemaps
import folium
from streamlit_folium import st_folium
import re
import time
from timezonefinder import TimezoneFinder
from dateutil import tz
import datetime
import numpy as np

def process_origins(df):
    """Process the origins DataFrame to add required columns and handle expansions"""
    if 'geoid' not in df.columns:
        df.insert(0, 'geoid', range(1, len(df) + 1))
    if 'Employee_Number' in df.columns:
        df.drop('Employee_Number', axis='columns', inplace=True)
    if 'count_employees' in df.columns:
        df = df.loc[df.index.repeat(df['count_employees'])].reset_index(drop=True)
        df.insert(0, 'Employee_ID', range(1, len(df) + 1), True)
    elif 'geoid' in df.columns and (df['geoid'].dtype != 'int64'):
        df.insert(0, 'Employee_ID', range(1, len(df) + 1), True)
    return df

def find_coordinate_columns(df, zipcode_data, is_destination=False):
    """Find and standardize coordinate columns in DataFrame"""
    col_names = sorted(df.filter(regex=r"(?i)^lat.*|^Y$|^lon.*|^X$").columns)
    
    if col_names:
        df.rename(columns={col_names[1]: 'Longitude', col_names[0]: 'Latitude'}, inplace=True)
        df['Coords'] = list(zip(df['Latitude'], df['Longitude']))
    else:
        zip_cols = df.filter(regex=r"(?i)postal|zip code.*|zipcode.*|zip*").columns.tolist()
        if zip_cols:
            df.rename(columns={zip_cols[0]: 'Zipcode'}, inplace=True)
            df['Zipcode'] = df['Zipcode'].astype(str).str.zfill(5)
            zipcode_data['STD_ZIP5'] = zipcode_data['STD_ZIP5'].astype(str).str.zfill(5)
            
            lat_col = zipcode_data.filter(regex=r"(?i)^lat.*|^Y$").columns[0]
            lon_col = zipcode_data.filter(regex=r"(?i)^lon.*|^X$").columns[0]
            
            df = df.merge(
                zipcode_data[['STD_ZIP5', lat_col, lon_col]],
                how='left', 
                left_on='Zipcode', 
                right_on='STD_ZIP5'
            )
            df.rename(columns={lat_col: 'Latitude', lon_col: 'Longitude'}, inplace=True)
            df.drop(columns=['STD_ZIP5'], inplace=True)
            df['Coords'] = list(zip(df['Latitude'], df['Longitude']))
    return df

def combine_address_fields(df, is_destination=False):
    """Combine address fields if coordinates are not present"""
    if 'Coords' not in df.columns:
        filter_pattern = r"address.*|city$|town$|state$|zip.*|Postal*" if is_destination else r"address.*|city$|town$|state$|zip code.*|zipcode.*|zip*"
        filter_df = df.filter(regex=re.compile(filter_pattern, re.IGNORECASE))
        df['ADDRESS_FULL'] = filter_df.apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
    return df

@st.cache_data(show_spinner=False)
def geocode_single_address(address: str, api_key: str) -> dict:
    """Geocode a single address with rate limiting"""
    gmaps = googlemaps.Client(key=api_key)
    try:
        geocode_result = gmaps.geocode(address)
        time.sleep(0.1)
        return geocode_result[0] if geocode_result else None
    except Exception as e:
        st.error(f'Geocoding error: {e}')
        return None

def geocode_addresses(addresses, api_key):
    """Geocode multiple addresses with progress bar"""
    results = []
    progress_bar = st.progress(0)
    for i, address in enumerate(addresses):
        result = geocode_single_address(address, api_key)
        if result:
            results.append({
                'input_string': address,
                'formatted_address': result.get('formatted_address'),
                'latitude': result['geometry']['location']['lat'],
                'longitude': result['geometry']['location']['lng'],
            })
        progress_bar.progress((i + 1) / len(addresses))
    progress_bar.empty()
    return pd.DataFrame(results)

def get_timezone_info(df):
    """Calculate timezone and departure time"""
    valid_coords = df[df['Latitude'].notna() & df['Longitude'].notna()]
    if len(valid_coords) == 0:
        st.error("No valid coordinates found")
        st.stop()
    
    first_coord = valid_coords.iloc[0]
    time_zone = TimezoneFinder().timezone_at(
        lng=float(first_coord['Longitude']), 
        lat=float(first_coord['Latitude'])
    )
    
    # Get current time in the target timezone
    tz_obj = tz.gettz(time_zone)
    current_time = datetime.datetime.now(tz_obj)
    
    # Calculate next Wednesday
    today = current_time.date()
    days_ahead = (2 - today.weekday() + 7) % 7  # 2 = Wednesday
    if days_ahead == 0:  # If today is Wednesday
        days_ahead = 7   # Go to next Wednesday
    
    next_wednesday = today + datetime.timedelta(days=days_ahead)
    departure_time = datetime.datetime.combine(next_wednesday, datetime.time(8, 0))
    departure_time = departure_time.replace(tzinfo=tz_obj)

    # If the calculated time is in the past, add another week
    if departure_time <= current_time:
        departure_time += datetime.timedelta(days=7)
    
    return time_zone, int(departure_time.timestamp())

@st.cache_data
def calculate_distance_matrix(origins_list, destinations_list, mode, departure_time, api_key):
    """Calculate distance matrix"""
    return googlemaps.Client(key=api_key).distance_matrix(
        origins=origins_list,
        destinations=destinations_list,
        mode=mode,
        units='imperial',
        departure_time=departure_time
    )

def get_map_center(coords_list):
    """Calculate map center coordinates"""
    valid_coords = [c for c in coords_list if not any(pd.isna(x) for x in c)]
    return [np.mean([c[0] for c in valid_coords]), np.mean([c[1] for c in valid_coords])] if valid_coords else [42.3601, -71.0589]

def main():
    st.title("Commute Impact Analysis")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar form
    with st.sidebar.form(key='main_form'):
        st.header("Input Parameters")
        origins_file = st.file_uploader("Upload Origins CSV", type=['csv'])
        destinations_file = st.file_uploader("Upload Destinations CSV", type=['csv'])
        method_transit = st.selectbox(
            'Transit Method',
            ('driving', 'transit')
        )
        submitted = st.form_submit_button('Run Analysis')

    # Process form submission
    if submitted:
        try:
            with st.spinner('Processing data...'):
                origins = pd.read_csv(origins_file)
                destinations = pd.read_csv(destinations_file)
                API_KEY = st.secrets["google_maps"]["api_key"]
                
                # Load ZIP code data
                try:
                    zipcode_data = pd.read_csv("ZIP_Code_Population_Weighted_Centroids.csv")
                except FileNotFoundError:
                    st.error("Missing ZIP code data file")
                    st.stop()

                # Process data
                origins = process_origins(origins)
                origins = find_coordinate_columns(origins, zipcode_data)
                origins = combine_address_fields(origins)
                destinations = find_coordinate_columns(destinations, zipcode_data, is_destination=True)
                destinations = combine_address_fields(destinations, is_destination=True)

                # Geocode missing coordinates
                if 'Coords' not in origins.columns:
                    origins = origins.merge(
                        geocode_addresses(origins['ADDRESS_FULL'].tolist(), API_KEY),
                        left_on='ADDRESS_FULL', 
                        right_on='input_string', 
                        how='left'
                    )
                    origins['Coords'] = list(zip(origins['latitude'], origins['longitude']))

                if 'Coords' not in destinations.columns:
                    destinations = destinations.merge(
                        geocode_addresses(destinations['ADDRESS_FULL'].tolist(), API_KEY),
                        left_on='ADDRESS_FULL', 
                        right_on='input_string', 
                        how='left'
                    )
                    destinations['Coords'] = list(zip(destinations['latitude'], destinations['longitude']))

                # Validate coordinates
                if origins['Coords'].isna().any() or destinations['Coords'].isna().any():
                    st.error("Missing coordinates in data")
                    st.stop()

                # Calculate matrix
                _, departure_time = get_timezone_info(origins)
                distance_matrix = calculate_distance_matrix(
                    origins['Coords'].tolist(),
                    destinations['Coords'].tolist(),
                    method_transit,
                    departure_time,
                    API_KEY
                )

                # Process results
                results = []
                for row in distance_matrix['rows']:
                    results.append([
                        (
                            element['duration']['value'] / 60,
                            element['distance']['value'] * 0.000621371
                        ) if element['status'] == 'OK' else (None, None)
                        for element in row['elements']
                    ])

                # Store in session state
                st.session_state.results = {
                    'origins': origins,
                    'destinations': destinations,
                    'durations': [[r[0] for r in row] for row in results],
                    'distances': [[r[1] for r in row] for row in results],
                    'map_center': get_map_center(
                        origins['Coords'].tolist() + destinations['Coords'].tolist()
                    )
                }

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            st.stop()

    # Display results if available
    if st.session_state.results:
        st.header("Analysis Results")
        
        # Create results dataframe
        durations_df = pd.DataFrame(
            st.session_state.results['durations'],
            columns=[f'Duration_to_{i}' for i in range(len(st.session_state.results['destinations']))]
        )

        durations_df = pd.DataFrame(
            st.session_state.results['durations'],
            columns=[f'Duration_to_{i}' for i in range(len(st.session_state.results['destinations']))]
        )
        distances_df = pd.DataFrame(
            st.session_state.results['distances'],
            columns=[f'Distance_to_{i+1}' for i in range(len(st.session_state.results['destinations']))]
        )
        results_df = pd.concat([
            st.session_state.results['origins'].reset_index(drop=True),
            durations_df,
            distances_df
        ], axis=1)

        # Show results
        st.dataframe(results_df.head())
        st.download_button(
            "Download Full Results",
            results_df.to_csv(index=False).encode('utf-8'),
            f"CommuteImpact_Results_{method_transit}_{datetime.date.today().strftime('%m%y')}.csv",
            "text/csv"
        )

        # Show map
        st.header("Map")
        map_obj = folium.Map(location=st.session_state.results['map_center'], tiles='stadiaalidadesmooth', zoom_start=8)
        
        for _, row in st.session_state.results['origins'].iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"Origin: {row.get('ADDRESS_FULL', '')}",
                # radius=3,
                # color='blue',
                # fill=True
            ).add_to(map_obj)
        
        for _, row in st.session_state.results['destinations'].iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"Destination: {row.get('ADDRESS_FULL', '')}",
                # radius=5,
                color='red',
                # fill=True
            ).add_to(map_obj)
        
        st_folium(map_obj, width=700, height=500)

if __name__ == "__main__":
    main()