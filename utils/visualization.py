import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import numpy as np
import streamlit as st

def get_map_center(coords_list):
    valid = [c for c in coords_list if not any(pd.isna(x) for x in c)]
    if valid:
        return [np.mean([c[0] for c in valid]), np.mean([c[1] for c in valid])]
    return [42.3601, -71.0589]  # fallback = Boston

def create_commute_map(origins_df, destinations_df, style_opts):
    # Map center
    all_coords = origins_df['Coords'].tolist() + destinations_df['Coords'].tolist()
    map_center = get_map_center(all_coords)

    # Base map
    m = folium.Map(location=map_center,
                   tiles=style_opts['tile_style'],
                   zoom_start=10,
                   control_scale=True)

    # Clusters
    origin_cluster = MarkerCluster(name="Origins").add_to(m) if style_opts['use_clustering'] else m
    dest_cluster = MarkerCluster(name="Destinations").add_to(m) if style_opts['use_clustering'] else m

    origin_coords, dest_coords = [], []

    # Origins
    for _, row in origins_df.iterrows():
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            continue
        coords = [row['Latitude'], row['Longitude']]
        origin_coords.append(coords)

        folium.CircleMarker(
            location=coords,
            radius=6,
            color='white',
            fill=True,
            fill_color=style_opts['origin_color'],
            popup=row.get('ADDRESS_FULL', 'Unknown')
        ).add_to(origin_cluster)

    # Destinations
    for _, row in destinations_df.iterrows():
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            continue
        coords = [row['Latitude'], row['Longitude']]
        dest_coords.append(coords)

        folium.Marker(
            location=coords,
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            popup=row.get('ADDRESS_FULL', 'Unknown')
        ).add_to(dest_cluster)

    # Connection lines
    if style_opts['show_connections']:
        for o in origin_coords:
            for d in dest_coords:
                folium.PolyLine([o, d],
                                color=style_opts['connection_color'],
                                weight=1,
                                opacity=0.5).add_to(m)

    # Heatmap (optional)
    if style_opts['show_heatmap'] and (origin_coords + dest_coords):
        HeatMap(origin_coords + dest_coords, radius=20).add_to(m)

    # Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; 
                background: white; padding: 10px; border-radius: 8px; 
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 14px;">
        <b>Legend</b><br>
        <span style="color:{style_opts['origin_color']}">●</span> Origins ({len(origin_coords)})<br>
        <span style="color:red">⚑</span> Destinations ({len(dest_coords)})
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m

def map_interface():
    st.sidebar.header("Map Options")
    return {
        'tile_style': st.sidebar.selectbox("Map Style",
                                           ['CartoDB dark_matter','CartoDB positron','OpenStreetMap'],
                                           0),
        'origin_color': st.sidebar.color_picker("Origin Color", "#00D4AA"),
        'connection_color': st.sidebar.color_picker("Connection Line", "#FFE66D"),def create_commute_map(origins_df, destinations_df, style_opts):
    # Collect coords from Lat/Lon columns
    origin_coords = origins_df.dropna(subset=["Latitude", "Longitude"])[["Latitude","Longitude"]].values.tolist()
    dest_coords   = destinations_df.dropna(subset=["Latitude", "Longitude"])[["Latitude","Longitude"]].values.tolist()

    # Map center
    map_center = get_map_center(origin_coords + dest_coords)

    # Base map
    m = folium.Map(location=map_center,
                   tiles=style_opts['tile_style'],
                   zoom_start=10,
                   control_scale=True)
    
    # Clusters
    origin_cluster = MarkerCluster(name="Origins").add_to(m) if style_opts['use_clustering'] else m
    dest_cluster   = MarkerCluster(name="Destinations").add_to(m) if style_opts['use_clustering'] else m

    # Origins
    for _, row in origins_df.iterrows():
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            continue
        coords = [row['Latitude'], row['Longitude']]
        folium.CircleMarker(
            location=coords,
            radius=6,
            color='white',
            fill=True,
            fill_color=style_opts['origin_color'],
            popup=row.get('ADDRESS_FULL', 'Unknown')
        ).add_to(origin_cluster)

    # Destinations
    for _, row in destinations_df.iterrows():
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            continue
        coords = [row['Latitude'], row['Longitude']]
        folium.Marker(
            location=coords,
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            popup=row.get('ADDRESS_FULL', 'Unknown')
        ).add_to(dest_cluster)

    # Connection lines
    if style_opts['show_connections']:
        for o in origin_coords:
            for d in dest_coords:
                folium.PolyLine([o, d],
                                color=style_opts['connection_color'],
                                weight=1,
                                opacity=0.5).add_to(m)

    # Heatmap
    if style_opts['show_heatmap'] and (origin_coords + dest_coords):
        HeatMap(origin_coords + dest_coords, radius=20).add_to(m)

    # Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; 
                background: white; padding: 10px; border-radius: 8px; 
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 14px;">
        <b>Legend</b><br>
        <span style="color:{style_opts['origin_color']}">●</span> Origins ({len(origin_coords)})<br>
        <span style="color:red">⚑</span> Destinations ({len(dest_coords)})
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m
