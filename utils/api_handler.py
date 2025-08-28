# utils/api_handler.py

import googlemaps
import streamlit as st
import time
from timezonefinder import TimezoneFinder
from dateutil import tz
import datetime
import numpy as np
import pandas as pd

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
                'Latitude': result['geometry']['location']['lat'],
                'Longitude': result['geometry']['location']['lng'],
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

def chunk_coordinates(coords_list, chunk_size=25):
    """Split coordinates into chunks to avoid MAX_ELEMENTS_EXCEEDED"""
    return [coords_list[i:i + chunk_size] for i in range(0, len(coords_list), chunk_size)]

def calculate_distance_matrix_in_chunks(origins_list, destinations_list, mode, departure_time, api_key):
    """Calculate distance matrix in chunks to handle API limits"""
    gmaps = googlemaps.Client(key=api_key)
    
    # Initialize progress bar
    total_operations = len(origins_list) * len(destinations_list)
    progress_bar = st.progress(0)
    operations_completed = 0
    
    # Chunk both origins and destinations - use smaller chunks to avoid MAX_ELEMENTS_EXCEEDED
    # Google Maps API has a limit of 100 elements per request (10x10 or 25x4, etc.)
    chunk_size = 10  # Reduced from 25 to 10
    origin_chunks = [origins_list[i:i + chunk_size] for i in range(0, len(origins_list), chunk_size)]
    destination_chunks = [destinations_list[i:i + chunk_size] for i in range(0, len(destinations_list), chunk_size)]
    
    # Initialize structure for results
    all_results = [[] for _ in range(len(origins_list))]
    
    try:
        for i, origin_chunk in enumerate(origin_chunks):
            origin_offset = i * chunk_size
            
            for j, dest_chunk in enumerate(destination_chunks):
                # Make API call for current chunks with error handling and retries
                try:
                    chunk_matrix = gmaps.distance_matrix(
                        origins=origin_chunk,
                        destinations=dest_chunk,
                        mode=mode,
                        units='imperial',
                        departure_time=departure_time
                    )
                    
                    # Extract results from current chunk
                    for k, row in enumerate(chunk_matrix['rows']):
                        origin_idx = origin_offset + k
                        if origin_idx < len(origins_list):  # Safety check
                            # Process each element in the row
                            for element in row['elements']:
                                if element['status'] == 'OK':
                                    all_results[origin_idx].append(
                                        (element['duration']['value'] / 60,  # Convert to minutes
                                            element['distance']['value'] * 0.000621371)  # Convert to miles
                                    )
                                else:
                                    all_results[origin_idx].append((None, None))
                    
                    # Update progress
                    operations_completed += len(origin_chunk) * len(dest_chunk)
                    progress_bar.progress(min(operations_completed / total_operations, 1.0))
                    
                except Exception as e:
                    st.warning(f"Error processing chunk {i}x{j}: {str(e)}. Retrying with smaller chunks...")
                    
                    # If we hit MAX_ELEMENTS_EXCEEDED, try with even smaller chunks
                    if "MAX_ELEMENTS_EXCEEDED" in str(e):
                        smaller_chunk_size = 5
                        sub_origins = [origin_chunk[x:x + smaller_chunk_size] for x in range(0, len(origin_chunk), smaller_chunk_size)]
                        sub_destinations = [dest_chunk[x:x + smaller_chunk_size] for x in range(0, len(dest_chunk), smaller_chunk_size)]
                        
                        for sub_i, sub_origin in enumerate(sub_origins):
                            for sub_j, sub_dest in enumerate(sub_destinations):
                                try:
                                    sub_matrix = gmaps.distance_matrix(
                                        origins=sub_origin,
                                        destinations=sub_dest,
                                        mode=mode,
                                        units='imperial',
                                        departure_time=departure_time
                                    )
                                    
                                    # Process this sub-chunk
                                    for k, row in enumerate(sub_matrix['rows']):
                                        origin_idx = origin_offset + (sub_i * smaller_chunk_size) + k
                                        if origin_idx < len(origins_list):
                                            results_offset = j * chunk_size + (sub_j * smaller_chunk_size)
                                            
                                            # Insert results at the right position
                                            for l, element in enumerate(row['elements']):
                                                insert_pos = results_offset + l
                                                result_val = (
                                                    (element['duration']['value'] / 60, element['distance']['value'] * 0.000621371)
                                                    if element['status'] == 'OK' else (None, None)
                                                )
                                                
                                                # Make sure we have enough space in the results list
                                                while len(all_results[origin_idx]) <= insert_pos:
                                                    all_results[origin_idx].append((None, None))
                                                    
                                                all_results[origin_idx][insert_pos] = result_val
                                                
                                except Exception as sub_e:
                                    st.error(f"Error in sub-chunk processing: {str(sub_e)}")
                
                # Add small delay to avoid hitting rate limits
                time.sleep(0.2)
        
        # Clear progress bar
        progress_bar.empty()
        
        # Convert to the distance matrix format expected by the rest of the code
        formatted_matrix = {
            'rows': [
                {
                    'elements': [
                        {
                            'duration': {'value': r[0] * 60 if r[0] is not None else None},
                            'distance': {'value': r[1] / 0.000621371 if r[1] is not None else None},
                            'status': 'OK' if r[0] is not None else 'NOT_FOUND'
                        }
                        for r in row
                    ]
                }
                for row in all_results
            ]
        }
        
        return formatted_matrix
        
    except Exception as e:
        st.error(f"Error during distance matrix calculation: {str(e)}")
        progress_bar.empty()
        return None
