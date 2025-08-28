# utils/visualization.py

import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
import pandas as pd

def get_map_center(coords_list):
    """Calculate map center coordinates"""
    valid_coords = [c for c in coords_list if not any(pd.isna(x) for x in c)]
    return [np.mean([c[0] for c in valid_coords]), np.mean([c[1] for c in valid_coords])] if valid_coords else [42.3601, -71.0589]

def create_commute_map(origins_df, destinations_df, map_center=None):
    """
    Creates a Folium map with origin and destination markers.
    """
    if map_center is None:
        all_coords = origins_df['Coords'].tolist() + destinations_df['Coords'].tolist()
        map_center = get_map_center(all_coords)

    map_obj = folium.Map(location=map_center, tiles='cartodbpositron', zoom_start=8)
    
    # Add origin markers
    for _, row in origins_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Origin: {row.get('ADDRESS_FULL', '')}",
            color='blue',
            radius=5
        ).add_to(map_obj)
    
    # Add destination markers
    for _, row in destinations_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Destination: {row.get('ADDRESS_FULL', '')}",
            color='red',
            radius=7
        ).add_to(map_obj)

    return map_obj
