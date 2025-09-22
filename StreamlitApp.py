# StreamlitApp.py

import streamlit as st
import pandas as pd
import datetime
from streamlit_folium import st_folium

from utils.data_processor import (
    process_origins,
    find_coordinate_columns,
    combine_address_fields,
    CommuteAnalyzer
)
from utils.api_handler import (
    geocode_addresses,
    get_timezone_info,
    calculate_distance_matrix_in_chunks
)
from utils.visualization import create_commute_map

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
        
        max_commute_time = st.slider(
            'Maximum Commute Time (minutes)',
            min_value=15, 
            max_value=120, 
            value=60, 
            step=5
        )

    if submitted:
        try:
            with st.spinner('Processing data...'):
                origins = pd.read_csv(origins_file)
                destinations = pd.read_csv(destinations_file)
                API_KEY = st.secrets["google_maps"]["api_key"]
                
                try:
                    zipcode_data = pd.read_csv("data/ZIP_Code_Population_Weighted_Centroids.csv")
                except FileNotFoundError:
                    st.error("Missing ZIP code data file. Please ensure it is in the 'data' folder.")
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
                }

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            st.stop()

    if st.session_state.results:
        st.header("Analysis Results")
        
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

        analyzer = CommuteAnalyzer({method_transit: raw_results_df})
        processed_df = analyzer.process_commute_data()
        transformed_df = analyzer.transform_for_visualization(processed_df, st.session_state.results['destinations'])

        filtered_df, total_employees, remaining_employees = analyzer.filter_by_commute_time(
            transformed_df, 
            max_commute_time
        )
        
        # Calculate averages for the map based on the filtered data
        filtered_durations = filtered_df[filtered_df['variable'].str.contains('CurrentCommute_Time')]['value']
        filtered_commute_times = pd.to_numeric(filtered_durations, errors='coerce')
        avg_duration = filtered_commute_times.mean()
        
        # NOTE: Calculating average distance requires a bit more logic.
        # Assuming for now we'll just use the same filtered group for a proxy
        filtered_distances = filtered_df[filtered_df['variable'].str.contains('Distance')]['value']
        avg_distance = pd.to_numeric(filtered_distances, errors='coerce').mean()
        
        st.subheader("Filtered Commute Data Summary")
        st.write(f"Total employees: {total_employees}")
        st.write(f"Employees meeting criteria: {remaining_employees}")
        
        st.subheader("Categorized Commute Times")
        st.dataframe(filtered_df.head())
        
        st.download_button(
            "Download Categorized Data",
            filtered_df.to_csv(index=False).encode('utf-8'),
            f"CommuteAnalysis_{method_transit}_{datetime.date.today().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

        st.header("Map Visualization")
        map_center = st.session_state.results.get('map_center')
        
        map_obj = create_commute_map(
            st.session_state.results['origins'],
            st.session_state.results['destinations'],
            map_center,
            # avg_duration,
            # avg_distance
        )
        st_folium(map_obj, width=700, height=500)

if __name__ == "__main__":
    main()
