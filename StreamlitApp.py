import streamlit as st
import pandas as pd
import datetime
from streamlit_folium import st_folium

from utils.data_processor import process_origins, find_coordinate_columns, combine_address_fields, CommuteAnalyzer
from utils.api_handler import geocode_addresses, get_timezone_info, calculate_distance_matrix_in_chunks
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

                # Process dataframes using functions from utils
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

                # Get timezone info and calculate distance matrix
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

                # Extract durations and distances
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
                    'distances': [[r[1] for r in row] for row in results]
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

        # Show map using the visualization utility
        st.header("Map Visualization")
        map_center = st.session_state.results['map_center'] if 'map_center' in st.session_state.results else None
        
        map_obj = create_commute_map(
            st.session_state.results['origins'],
            st.session_state.results['destinations'],
            map_center
        )
        st_folium(map_obj, width=700, height=500)

if __name__ == "__main__":
    main()
