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
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class CommuteData:
    """Data class to store commute analysis results"""
    dataframes: List[pd.DataFrame]
    dataframes_dict: Dict[str, pd.DataFrame]
    split_files: List[str]

class CommuteAnalyzer:
    """Class to handle commute data analysis and transformation"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.commute_data = self._structure_data(data_dict)
        
    def _structure_data(self, data_dict: Dict[str, pd.DataFrame]) -> CommuteData:
        """Structure in-memory data into CommuteData format"""
        dfs = []
        split_f = []
        dicty = {}
        
        for method, df in data_dict.items():
            dfs.append(df)
            dicty[method] = df
            split_f.append([method])
            
        return CommuteData(dfs, dicty, split_f)
    
    @staticmethod
    def _commute_time_bucket(time: float) -> str:
        """Categorize commute time into buckets"""
        if time == 0:
            return '0'
        elif time <= 15:
            return '0-15 Minutes'
        elif 15 < time <= 30:
            return '15-30 Minutes'
        elif 30 < time <= 45:
            return '30-45 Minutes'
        elif 45 < time <= 60:
            return '45-60 Minutes'
        else:
            return '60 Minutes +'
    
    @staticmethod
    def _calculate_time_change(time_diff: float) -> int:
        """Calculate time change bucket"""
        if time_diff <= -20:
            return -25
        elif -20 < time_diff <= -15:
            return -20
        elif -15 < time_diff <= -10:
            return -15
        elif -10 < time_diff <= -5:
            return -5
        elif -5 < time_diff < 0:
            return -5
        elif 0 < time_diff <= 5:
            return 5
        elif 5 < time_diff <= 10:
            return 10
        elif 10 < time_diff <= 15:
            return 15
        elif 15 < time_diff <= 20:
            return 20
        elif time_diff > 20:
            return 25
        return 0
    
    @staticmethod
    def _determine_time_change(time_diff: float) -> str:
        """Determine if time was reduced, added, or unchanged"""
        if time_diff < 0:
            return "Time Reduced"
        elif time_diff > 0:
            return "Time Added"
        return "No Change"
    
    def process_commute_data(self) -> pd.DataFrame:
        """Process and merge commute data"""
        dataframes_dict = self.commute_data.dataframes_dict
        methods = list(dataframes_dict.keys())
        
        cleaned_dfs = []
        for method in methods:
            df = dataframes_dict[method].copy()
            df.insert(1, 'Method', method)
            cleaned_dfs.append(df)
        
        return pd.concat(cleaned_dfs).reset_index(drop=True)
    
    def transform_for_visualization(self, df: pd.DataFrame, destinations_df: pd.DataFrame) -> pd.DataFrame:
        """Transform data for visualization in the target format"""
        # Identify duration columns
        duration_cols = [col for col in df.columns if col.startswith('Duration')]
        base_cols = ['Employee_ID', 'Method', 'Zipcode', 'Latitude', 'Longitude', 'ADDRESS_FULL']
        
        # Get destination addresses
        destination_addresses = destinations_df['ADDRESS_FULL'].tolist()
        
        # Initialize list to store all transformations
        transformed_data = []
        
        # Process each employee's data
        for _, row in df.iterrows():
            employee_base = {col: row[col] for col in base_cols}
            current_commute = row[duration_cols[0]]  # Assuming first duration is current
            current_dest_address = destination_addresses[0]  # First destination for current commute
            
            # Current commute data
            transformed_data.append({
                **employee_base,
                'variable': 'CurrentCommute_Time',
                'value': str(current_commute),
                'names': current_dest_address
            })
            
            transformed_data.append({
                **employee_base,
                'variable': 'Current_Commute_Time_Bucket',
                'value': self._commute_time_bucket(current_commute),
                'names': current_dest_address
            })
            
            # Process each potential location
            for i, duration_col in enumerate(duration_cols[1:], 1):
                commute_time = row[duration_col]
                time_diff = commute_time - current_commute
                dest_address = destination_addresses[i]  # Get corresponding destination address
                
                # Potential location commute time
                transformed_data.append({
                    **employee_base,
                    'variable': f'Potential_Location_{i}',
                    'value': str(commute_time),
                    'names': dest_address
                })
                
                # Time reduction bucket
                transformed_data.append({
                    **employee_base,
                    'variable': f'Potential_Commute_Time_Reduced_Bucket_{i}',
                    'value': str(self._calculate_time_change(time_diff)),
                    'names': dest_address
                })
                
                # Change in commute
                transformed_data.append({
                    **employee_base,
                    'variable': f'Change_Commute_{i}',
                    'value': str(time_diff),
                    'names': dest_address
                })
                
                # Time reduction category
                transformed_data.append({
                    **employee_base,
                    'variable': f'Commute_Time_Reduced_Bucket_{i}',
                    'value': str(self._calculate_time_change(time_diff)),
                    'names': dest_address
                })
                
                # Time change category
                transformed_data.append({
                    **employee_base,
                    'variable': f'Commute_Time_Category_Bucket_{i}',
                    'value': self._determine_time_change(time_diff),
                    'names': dest_address
                })
        
        # Convert to DataFrame and order columns
        result_df = pd.DataFrame(transformed_data)
        result_df = result_df[['Employee_ID', 'Method', 'Latitude', 'Longitude', 'Zipcode', 'variable', 'value', 'names']]
        
        return result_df

# Original Streamlit App Functions (slightly modified)
def process_origins(df):
    """Process the origins DataFrame"""
    # Ensure Employee_ID exists
    if 'Employee_ID' not in df.columns:
        df.insert(0, 'Employee_ID', range(1, len(df) + 1))
    
    # Handle geoid column
    if 'geoid' not in df.columns:
        df.insert(1, 'geoid', range(1, len(df) + 1))
    
    # Handle employee expansion
    if 'count_employees' in df.columns:
        df = df.loc[df.index.repeat(df['count_employees'])].reset_index(drop=True)
        df['Employee_ID'] = range(1, len(df) + 1)
    
    # Rename geoid to match analysis expectations
    df.rename(columns={'geoid': 'Geoid'}, inplace=True)
    
    return df

def find_coordinate_columns(df, zipcode_data, is_destination=False):
    """Find and standardize coordinate columns"""
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
    """Combine address fields if coordinates missing"""
    # Always create ADDRESS_FULL
    filter_pattern = r"address.*|city$|town$|state$|zip.*|Postal*" if is_destination else r"address.*|city$|town$|state$|zip code.*|zipcode.*|zip*"
    filter_df = df.filter(regex=re.compile(filter_pattern, re.IGNORECASE))
    df['ADDRESS_FULL'] = filter_df.apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
    return df

@st.cache_data(show_spinner=False)
def geocode_single_address(address: str, api_key: str) -> dict:
    """Geocode single address"""
    gmaps = googlemaps.Client(key=api_key)
    try:
        geocode_result = gmaps.geocode(address)
        time.sleep(0.1)
        return geocode_result[0] if geocode_result else None
    except Exception as e:
        st.error(f'Geocoding error: {e}')
        return None

def geocode_addresses(addresses, api_key):
    """Geocode multiple addresses"""
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
    
    tz_obj = tz.gettz(time_zone)
    current_time = datetime.datetime.now(tz_obj)
    
    today = current_time.date()
    days_ahead = (2 - today.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    
    next_wednesday = today + datetime.timedelta(days=days_ahead)
    departure_time = datetime.datetime.combine(next_wednesday, datetime.time(8, 0))
    departure_time = departure_time.replace(tzinfo=tz_obj)

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
    
    if 'results' not in st.session_state:
        st.session_state.results = None

    with st.sidebar.form(key='main_form'):
        st.header("Input Parameters")
        origins_file = st.file_uploader("Upload Origins CSV", type=['csv'])
        destinations_file = st.file_uploader("Upload Destinations CSV", type=['csv'])
        method_transit = st.selectbox('Transit Method', ('driving', 'transit'))
        submitted = st.form_submit_button('Run Analysis')

    if submitted:
        try:
            with st.spinner('Processing data...'):
                origins = pd.read_csv(origins_file)
                destinations = pd.read_csv(destinations_file)
                API_KEY = st.secrets["google_maps"]["api_key"]
                
                try:
                    zipcode_data = pd.read_csv("ZIP_Code_Population_Weighted_Centroids.csv")
                except FileNotFoundError:
                    st.error("Missing ZIP code data file")
                    st.stop()

                origins = process_origins(origins)
                origins = find_coordinate_columns(origins, zipcode_data)
                origins = combine_address_fields(origins)
                destinations = find_coordinate_columns(destinations, zipcode_data, is_destination=True)
                destinations = combine_address_fields(destinations, is_destination=True)

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

                if origins['Coords'].isna().any() or destinations['Coords'].isna().any():
                    st.error("Missing coordinates in data")
                    st.stop()

                def chunk_coordinates(coords_list, chunk_size=25):
                    """Split coordinates into chunks to avoid MAX_ELEMENTS_EXCEEDED"""
                    return [coords_list[i:i + chunk_size] for i in range(0, len(coords_list), chunk_size)]

                def calculate_distance_matrix_in_chunks(origins_list, destinations_list, mode, departure_time, api_key):
                    """Calculate distance matrix in chunks to handle API limits"""
                    gmaps = googlemaps.Client(key=api_key)
                    
                    # Initialize progress bar
                    total_chunks = len(origins_list) * len(destinations_list)
                    progress_bar = st.progress(0)
                    chunk_counter = 0
                    
                    # Chunk both origins and destinations
                    origin_chunks = chunk_coordinates(origins_list)
                    destination_chunks = chunk_coordinates(destinations_list)
                    
                    all_results = []
                    
                    try:
                        for origin_chunk in origin_chunks:
                            chunk_results = []
                            
                            for dest_chunk in destination_chunks:
                                # Make API call for current chunks
                                chunk_matrix = gmaps.distance_matrix(
                                    origins=origin_chunk,
                                    destinations=dest_chunk,
                                    mode=mode,
                                    units='imperial',
                                    departure_time=departure_time
                                )
                                
                                # Extract results from current chunk
                                for row in chunk_matrix['rows']:
                                    chunk_results.append([
                                        (
                                            element['duration']['value'] / 60,
                                            element['distance']['value'] * 0.000621371
                                        ) if element['status'] == 'OK' else (None, None)
                                        for element in row['elements']
                                    ])
                                
                                # Update progress
                                chunk_counter += 1
                                progress_bar.progress(chunk_counter / total_chunks)
                                
                                # Add small delay to avoid hitting rate limits
                                time.sleep(0.1)
                            
                            all_results.extend(chunk_results)
                        
                        # Clear progress bar
                        progress_bar.empty()
                        
                        # Combine results
                        combined_results = []
                        current_row = []
                        
                        for row in all_results:
                            current_row.extend(row)
                            if len(current_row) == len(destinations_list):
                                combined_results.append(current_row)
                                current_row = []
                        
                        return {
                            'rows': [
                                {'elements': [
                                    {
                                        'duration': {'value': r[0] * 60},
                                        'distance': {'value': r[1] / 0.000621371},
                                        'status': 'OK' if r[0] is not None else 'NOT_FOUND'
                                    }
                                    for r in row
                                ]}
                                for row in combined_results
                            ]
                        }
                        
                    except Exception as e:
                        st.error(f"Error during distance matrix calculation: {str(e)}")
                        progress_bar.empty()
                        return None

                _, departure_time = get_timezone_info(origins)
                distance_matrix = calculate_distance_matrix_in_chunks(
                    origins['Coords'].tolist(),
                    destinations['Coords'].tolist(),
                    method_transit,
                    departure_time,
                    API_KEY
                )

                if distance_matrix is None:
                    st.error("Failed to calculate distance matrix")
                    st.stop()

                results = []
                for row in distance_matrix['rows']:
                    results.append([
                        (
                            element['duration']['value'] / 60,
                            element['distance']['value'] * 0.000621371
                        ) if element['status'] == 'OK' else (None, None)
                        for element in row['elements']
                    ])

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

    if st.session_state.results:
        st.header("Analysis Results")
        
        # Create raw results dataframe
        durations_df = pd.DataFrame(
            st.session_state.results['durations'],
            columns=[f'Duration_to_{i}' for i in range(len(st.session_state.results['destinations']))]
        )
        
        distances_df = pd.DataFrame(
            st.session_state.results['distances'],
            columns=[f'Distance_to_{i+1}' for i in range(len(st.session_state.results['destinations']))]
        )
        
        raw_results_df = pd.concat([
            st.session_state.results['origins'].reset_index(drop=True),
            durations_df,
            distances_df
        ], axis=1)

        # Process with CommuteAnalyzer
        analyzer = CommuteAnalyzer({method_transit: raw_results_df})
        processed_df = analyzer.process_commute_data()
        final_df = analyzer.transform_for_visualization(processed_df, st.session_state.results['destinations'])

        # Show processed data
        st.subheader("Categorized Commute Times")
        st.dataframe(final_df.head())
        
        st.download_button(
            "Download Categorized Data",
            final_df.to_csv(index=False).encode('utf-8'),
            f"CommuteAnalysis_{method_transit}_{datetime.date.today().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

        # Show map
        st.header("Map Visualization")
        map_obj = folium.Map(location=st.session_state.results['map_center'], tiles='stadiaalidadesmooth', zoom_start=8)
        
        for _, row in st.session_state.results['origins'].iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"Origin: {row.get('ADDRESS_FULL', '')}",
                color='blue',
                radius=5
            ).add_to(map_obj)
        
        for _, row in st.session_state.results['destinations'].iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"Destination: {row.get('ADDRESS_FULL', '')}",
                color='red',
                radius=7
            ).add_to(map_obj)
        
        st_folium(map_obj, width=700, height=500)

if __name__ == "__main__":
    main()