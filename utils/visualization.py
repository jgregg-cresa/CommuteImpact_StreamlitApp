# utils/visualization.py

import folium
from folium.plugins import MarkerCluster
import numpy as np
import pandas as pd

def get_map_center(coords_list):
    """Calculate map center coordinates"""
    valid_coords = [c for c in coords_list if not any(pd.isna(x) for x in c)]
    return [np.mean([c[0] for c in valid_coords]), np.mean([c[1] for c in valid_coords])] if valid_coords else [42.3601, -71.0589]

def create_commute_map(origins_df, destinations_df, map_center=None):
    """
    Creates a Folium map with clustered origin markers,
    destination markers, and a clean legend.
    """
    if map_center is None:
        all_coords = origins_df['Coords'].tolist() + destinations_df['Coords'].tolist()
        map_center = get_map_center(all_coords)

    map_obj = folium.Map(location=map_center, tiles='cartodbpositron', zoom_start=8)

    # --- Origin clusters ---
    origin_cluster = MarkerCluster(name="Origins").add_to(map_obj)
    for _, row in origins_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Origin: {row.get('ADDRESS_FULL', '')}",
            color='blue',
            fill=True,
            fill_opacity=0.7,
            radius=5
        ).add_to(origin_cluster)

    # --- Destination markers ---
    for _, row in destinations_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Destination: {row.get('ADDRESS_FULL', '')}",
            color='red',
            fill=True,
            fill_opacity=0.8,
            radius=7
        ).add_to(map_obj)

    # --- Legend (HTML overlay) ---
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 20px; left: 20px; width: 140px; 
        background: white; padding: 8px; 
        border:2px solid grey; border-radius:8px;
        font-size:14px; z-index:9999;
    ">
    <b>Legend</b><br>
    <span style="color:blue;">●</span> Origins<br>
    <span style="color:red;">●</span> Destinations
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))

    return map_obj
