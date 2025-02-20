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

def process_origins(df):
    """Process the origins DataFrame to add required columns and handle expansions"""
    # Add geoid if not present
    if 'geoid' not in df.columns:
        df.insert(0, 'geoid', range(1, len(df) + 1))
        st.write("Added 'geoid' column to origins.")

    # Drop Employee_Number if present
    if 'Employee_Number' in df.columns:
        df.drop('Employee_Number', axis='columns', inplace=True)
        st.write("Dropped 'Employee_Number' column from origins.")

    # Handle count_employees expansion
    if 'count_employees' in df.columns:
        df = df.loc[df.index.repeat(df['count_employees'])].reset_index(drop=True)
        df.insert(0, 'Employee_ID', range(1, len(df) + 1), True)
        st.write("Expanded origins based on 'count_employees' and added 'Employee_ID' column.")
    elif 'geoid' in df.columns and (df['geoid'].dtype != 'int64'):
        df.insert(0, 'Employee_ID', range(1, len(df) + 1), True)
        st.write("Added 'Employee_ID' column to origins.")

    return df

def find_coordinate_columns(df, is_destination=False):
    """Find and standardize coordinate columns in DataFrame"""
    filter_df = df.filter(regex=re.compile(r"^lat.*|^Y$|^lon.*|^X$", re.IGNORECASE))
    col_names = list(filter_df.columns)
    col_names.sort()
    
    if len(col_names) > 0:
        df = df.rename(columns={col_names[1]: 'Longitude', col_names[0]: 'Latitude'})
        df['Coords'] = list(zip(df['Latitude'], df['Longitude']))
        st.write(f"{'Destinations' if is_destination else 'Origins'} dataset contains latitude and longitude.")
    else:
        if not is_destination:
            # Handle zip codes for origins
            filter_df_zips = df.filter(regex=re.compile(r"postal|zip code.*|zipcode.*|zip*", re.IGNORECASE))
            zip_cols = list(filter_df_zips.columns)
            if len(zip_cols) > 0:
                df = df.rename(columns={zip_cols[0]: 'Zipcode'})
                st.write("Zip codes found in origins.")
            else:
                st.error("No latitude/longitude or zip codes found in origins.")
                st.stop()
        else:
            st.write("Destinations dataset does not contain latitude and longitude. Geocoding will be performed.")
    
    return df

def combine_address_fields(df, is_destination=False):
    """Combine address fields if coordinates are not present"""
    if 'Coords' not in df.columns:
        filter_pattern = r"address.*|city$|town$|state$|zip.*|Postal*" if is_destination else r"address.*|city$|town$|state$|zip code.*|zipcode.*|zip*"
        filter_df = df.filter(regex=re.compile(filter_pattern, re.IGNORECASE))
        df['ADDRESS_FULL'] = filter_df.apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
        st.write(f"Combined address fields for {'destinations' if is_destination else 'origins'}.")
    else:
        st.write(f"{'Destinations' if is_destination else 'Origins'} dataset is pre-geocoded.")
    return df

@st.cache_data(show_spinner=False)
def geocode_single_address(address: str, api_key: str) -> dict:
    """Geocode a single address using Google Maps API"""
    gmaps = googlemaps.Client(key=api_key)
    try:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            result = geocode_result[0]
            return {
                'input_string': address,
                'formatted_address': result.get('formatted_address'),
                'latitude': result['geometry']['location']['lat'],
                'longitude': result['geometry']['location']['lng'],
                'location_type': result['geometry']['location_type'],
                'place_id': result.get('place_id'),
            }
        return {
            'input_string': address,
            'formatted_address': None,
            'latitude': None,
            'longitude': None,
            'location_type': None,
            'place_id': None,
        }
    except Exception as e:
        st.error(f'Geocoding error for address {address}: {e}')
        return None

def geocode_addresses(addresses, api_key):
    """Geocode multiple addresses"""
    results = []
    progress_bar = st.progress(0)
    for i, address in enumerate(addresses):
        result = geocode_single_address(address, api_key)
        if result:
            results.append(result)
        progress_bar.progress((i + 1) / len(addresses))
    progress_bar.empty()
    return pd.DataFrame(results)

def get_timezone_info(df):
    """Calculate timezone and departure time based on first valid coordinates"""
    # Ensure we have valid coordinates
    valid_coords = df[df['latitude'].notna() & df['longitude'].notna()].iloc[0]
    obj = TimezoneFinder()
    time_zone = obj.timezone_at(lng=float(valid_coords['longitude']), 
                              lat=float(valid_coords['latitude']))
    
    # # Calculate departure time
    # setttt = tz.gettz(time_zone)
    # today = datetime.date.today()
    # offset = (today.weekday() - 2) % 7
    # last_wednesday = today - datetime.timedelta(days=offset)
    # departure_time = datetime.datetime.combine(last_wednesday, datetime.time(8, 0))
    # departure_time = departure_time.astimezone(setttt)
    
    # return time_zone, int(departure_time.timestamp())
    
    # Calculate departure time for next Wednesday
    setttt = tz.gettz(time_zone)
    today = datetime.date.today()
    days_ahead = (2 - today.weekday() + 7) % 7  # 2 = Wednesday
    if days_ahead == 0:  # If today is Wednesday, go to next Wednesday
        days_ahead = 7
    next_wednesday = today + datetime.timedelta(days=days_ahead)
    departure_time = datetime.datetime.combine(next_wednesday, datetime.time(8, 0))
    departure_time = departure_time.astimezone(setttt)
    
    return time_zone, int(departure_time.timestamp())

@st.cache_data
def calculate_distance_matrix(origins_list, destinations_list, mode, departure_time, api_key):
    """Calculate distance matrix using Google Maps API"""
    gmaps = googlemaps.Client(key=api_key)
    return gmaps.distance_matrix(
        origins=origins_list,
        destinations=destinations_list,
        mode=mode,
        units='imperial',
        departure_time=departure_time
    )

def main():
    st.title("Commute Impact Analysis")
    
    # File uploaders
    origins_file = st.sidebar.file_uploader("Upload Origins CSV", type=['csv'])
    destinations_file = st.sidebar.file_uploader("Upload Destinations CSV", type=['csv'])
    
    # Transit method selection
    method_transit = st.sidebar.selectbox(
        'Select transit method',
        ('driving', 'transit', 'walking', 'bicycling')
    )
    
    if st.sidebar.button('Run Analysis'):
        if not (origins_file and destinations_file):
            st.error('Please upload both Origins and Destinations CSV files.')
            return
            
        try:
            st.write('Processing data...')
            
            # Read files
            origins = pd.read_csv(origins_file)
            destinations = pd.read_csv(destinations_file)
            
            # Get API key
            API_KEY = st.secrets["google_maps"]["api_key"]
            
            # Process DataFrames
            origins = process_origins(origins)
            origins = find_coordinate_columns(origins)
            destinations = find_coordinate_columns(destinations, is_destination=True)
            
            origins = combine_address_fields(origins)
            destinations = combine_address_fields(destinations, is_destination=True)
            
            # Geocode if needed
            if 'Coords' not in origins.columns:
                st.write("Geocoding origins...")
                geocode_results = geocode_addresses(origins['ADDRESS_FULL'].tolist(), API_KEY)
                origins = origins.merge(geocode_results, left_on='ADDRESS_FULL', 
                                     right_on='input_string', how='left')
                origins['Coords'] = list(zip(origins['latitude'], origins['longitude']))
            
            if 'Coords' not in destinations.columns:
                st.write("Geocoding destinations...")
                geocode_results_dest = geocode_addresses(destinations['ADDRESS_FULL'].tolist(), API_KEY)
                destinations = destinations.merge(geocode_results_dest, 
                                               left_on='ADDRESS_FULL', 
                                               right_on='input_string', how='left')
                destinations['Coords'] = list(zip(destinations['latitude'], 
                                               destinations['longitude']))
            
            # Calculate timezone and departure time
            time_zone, departure_time = get_timezone_info(origins)
            
            # Calculate distance matrix
            st.write("Calculating distances and durations...")
            distance_matrix = calculate_distance_matrix(
                origins['Coords'].tolist(),
                destinations['Coords'].tolist(),
                method_transit,
                departure_time,
                API_KEY
            )
            
            # Process results
            durations = []
            distances = []
            for row in distance_matrix['rows']:
                row_durations = []
                row_distances = []
                for element in row['elements']:
                    if element['status'] == 'OK':
                        row_durations.append(element['duration']['value'] / 60)  # minutes
                        row_distances.append(element['distance']['value'] * 0.000621371)  # miles
                    else:
                        row_durations.append(None)
                        row_distances.append(None)
                durations.append(row_durations)
                distances.append(row_distances)
            
            # Create results DataFrame
            durations_df = pd.DataFrame(durations, 
                                      columns=[f'Duration_to_{i+1}' 
                                              for i in range(len(destinations))])
            distances_df = pd.DataFrame(distances, 
                                      columns=[f'Distance_to_{i+1}' 
                                              for i in range(len(destinations))])
            results_df = pd.concat([origins.reset_index(drop=True), 
                                  durations_df, distances_df], axis=1)
            
            # Display results
            st.subheader("Sample Results:")
            st.write(results_df.head())
            
            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results as CSV",
                csv,
                "commute_impact_results.csv",
                "text/csv"
            )
            
            # Map visualization
            st.subheader("Map Visualization")
            map1 = folium.Map(location=origins['Coords'].iloc[0], zoom_start=10)
            
            # Add markers
            for idx, row in origins.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"Origin: {row.get('ADDRESS_FULL', '')}",
                    radius=3,
                    color='blue',
                    fill=True,
                    fill_color='blue'
                ).add_to(map1)
                
            for idx, row in destinations.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"Destination: {row.get('ADDRESS_FULL', '')}",
                    radius=5,
                    color='red',
                    fill=True,
                    fill_color='red'
                ).add_to(map1)
            
            st_folium(map1, width=700, height=500)
            st.success('Data processed successfully.')
            
        except Exception as e:
            st.error(f'An error occurred: {e}')
            st.exception(e)

if __name__ == "__main__":
    main()