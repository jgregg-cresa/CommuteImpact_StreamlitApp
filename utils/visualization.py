import folium
from folium.plugins import HeatMap, MarkerCluster, BeautifyIcon, FastMarkerCluster
import numpy as np
import pandas as pd
import streamlit as st

def get_map_center(coords_list):
    """Calculate map center coordinates with better handling"""
    valid_coords = [c for c in coords_list if not any(pd.isna(x) for x in c)]
    if valid_coords:
        return [np.mean([c[0] for c in valid_coords]), np.mean([c[1] for c in valid_coords])]
    return [42.3601, -71.0589]  # Default to Boston

def create_commute_map(origins_df, destinations_df, map_center=None, style_options=None):
    """
    Creates a visually enhanced Folium map with modern styling and interactive features.
    """
    # Default style options
    default_style = {
        'tile_style': 'CartoDB dark_matter',
        'origin_color': '#00D4AA',  # Bright teal
        'destination_color': '#FF6B6B',  # Coral red
        'connection_color': '#FFE66D',  # Golden yellow
        'show_connections': True,
        'use_clustering': True,
        'show_heatmap': False,
        'animation': True
    }
    
    if style_options:
        default_style.update(style_options)
    
    if map_center is None:
        all_coords = origins_df['Coords'].tolist() + destinations_df['Coords'].tolist()
        map_center = get_map_center(all_coords)
    
    # Create map with modern dark theme
    map_obj = folium.Map(
        location=map_center, 
        tiles=default_style['tile_style'],
        zoom_start=10,
        prefer_canvas=True,
        control_scale=True
    )
    
    # Add custom CSS for enhanced styling
    map_obj.get_root().html.add_child(folium.Element("""
    <style>
        .leaflet-popup-content-wrapper {
            background: rgba(30, 30, 30, 0.95);
            color: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }
        .leaflet-popup-tip {
            background: rgba(30, 30, 30, 0.95);
        }
        .pulse-marker {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }
    </style>
    """))
    
    # Create marker clusters for better performance with many points
    if default_style['use_clustering']:
        origin_cluster = MarkerCluster(
            name="Origins",
            options={'maxClusterRadius': 50, 'spiderfyOnMaxZoom': True}
        ).add_to(map_obj)
        destination_cluster = MarkerCluster(
            name="Destinations", 
            options={'maxClusterRadius': 50, 'spiderfyOnMaxZoom': True}
        ).add_to(map_obj)
    
    # Enhanced origin markers with custom icons
    origin_coords = []
    for idx, row in origins_df.iterrows():
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            continue
            
        coords = [row['Latitude'], row['Longitude']]
        origin_coords.append(coords)
        
        # Create enhanced popup with more information
        popup_html = f"""
        <div style="font-family: 'Segoe UI', sans-serif; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: {default_style['origin_color']};">
                🏠 Origin Location
            </h4>
            <p style="margin: 5px 0;"><b>Address:</b><br>{row.get('ADDRESS_FULL', 'Unknown')}</p>
            <p style="margin: 5px 0;"><b>Coordinates:</b><br>{coords[0]:.4f}, {coords[1]:.4f}</p>
        </div>
        """
        
        marker = folium.CircleMarker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Origin: {row.get('ADDRESS_FULL', 'Unknown')[:50]}...",
            color='white',
            fillColor=default_style['origin_color'],
            weight=2,
            radius=8,
            fillOpacity=0.8,
            className='pulse-marker' if default_style['animation'] else ''
        )
        
        if default_style['use_clustering']:
            marker.add_to(origin_cluster)
        else:
            marker.add_to(map_obj)
    
    # Enhanced destination markers
    destination_coords = []
    for idx, row in destinations_df.iterrows():
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            continue
            
        coords = [row['Latitude'], row['Longitude']]
        destination_coords.append(coords)
        
        popup_html = f"""
        <div style="font-family: 'Segoe UI', sans-serif; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: {default_style['destination_color']};">
                🎯 Destination
            </h4>
            <p style="margin: 5px 0;"><b>Address:</b><br>{row.get('ADDRESS_FULL', 'Unknown')}</p>
            <p style="margin: 5px 0;"><b>Coordinates:</b><br>{coords[0]:.4f}, {coords[1]:.4f}</p>
        </div>
        """
        
        marker = folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Destination: {row.get('ADDRESS_FULL', 'Unknown')[:50]}...",
            icon=folium.Icon(
                color='red', 
                icon='flag',
                prefix='fa'
            )
        )
        
        if default_style['use_clustering']:
            marker.add_to(destination_cluster)
        else:
            marker.add_to(map_obj)
    
    # Add connection lines between origins and destinations
    if default_style['show_connections'] and origin_coords and destination_coords:
        for orig in origin_coords:
            for dest in destination_coords:
                folium.PolyLine(
                    locations=[orig, dest],
                    color=default_style['connection_color'],
                    weight=2,
                    opacity=0.6,
                    dash_array='5, 10'
                ).add_to(map_obj)
    
    # Add heatmap overlay (optional)
    if default_style['show_heatmap']:
        all_points = origin_coords + destination_coords
        if all_points:
            HeatMap(
                all_points,
                radius=20,
                blur=15,
                gradient={'0.2': 'blue', '0.4': 'cyan', '0.6': 'lime', '0.8': 'yellow', '1.0': 'red'}
            ).add_to(map_obj)
    
    # Add layer control
    folium.LayerControl().add_to(map_obj)
    
    # Add a legend
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background: rgba(30, 30, 30, 0.9); border-radius: 10px;
                border: 2px solid rgba(255, 255, 255, 0.2); z-index:9999; 
                font-size: 14px; color: white; padding: 15px;">
        <h4 style="margin: 0 0 10px 0; text-align: center;">Map Legend</h4>
        <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:{default_style['origin_color']}"></i> Origins</p>
        <p style="margin: 5px 0;"><i class="fa fa-flag" style="color:{default_style['destination_color']}"></i> Destinations</p>
        {f'<p style="margin: 5px 0;"><i class="fa fa-minus" style="color:{default_style["connection_color"]}"></i> Connections</p>' if default_style['show_connections'] else ''}
        <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.3);">
        <p style="margin: 0; font-size: 12px; opacity: 0.8;">
            Origins: {len(origin_coords)}<br>
            Destinations: {len(destination_coords)}
        </p>
    </div>
    '''
    map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    return map_obj

# Streamlit UI for customization
def create_map_interface():
    """Create Streamlit interface for map customization"""
    st.title("🗺️ Enhanced Commute Map Visualization")
    
    # Sidebar for customization options
    with st.sidebar:
        st.header("🎨 Customization Options")
        
        tile_style = st.selectbox(
            "Map Style",
            ['CartoDB dark_matter', 'CartoDB positron', 'OpenStreetMap', 'Stamen Terrain'],
            index=0
        )
        
        origin_color = st.color_picker("Origin Color", "#00D4AA")
        destination_color = st.color_picker("Destination Color", "#FF6B6B")
        connection_color = st.color_picker("Connection Color", "#FFE66D")
        
        show_connections = st.checkbox("Show Connection Lines", True)
        use_clustering = st.checkbox("Use Marker Clustering", True)
        show_heatmap = st.checkbox("Show Heatmap Overlay", False)
        animation = st.checkbox("Animated Markers", True)
    
    style_options = {
        'tile_style': tile_style,
        'origin_color': origin_color,
        'destination_color': destination_color,
        'connection_color': connection_color,
        'show_connections': show_connections,
        'use_clustering': use_clustering,
        'show_heatmap': show_heatmap,
        'animation': animation
    }
    
    return style_options

# Example usage function
def main(origins_df, destinations_df):
    """Main function to create and display the enhanced map"""
    
    # Get customization options from Streamlit interface
    style_options = create_map_interface()
    
    # Create the enhanced map
    enhanced_map = create_enhanced_commute_map(
        origins_df, 
        destinations_df, 
        style_options=style_options
    )
    
    # Display the map in Streamlit
    st.subheader("📍 Interactive Commute Map")
    
    # Add some metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Origins", len(origins_df))
    with col2:
        st.metric("Destinations", len(destinations_df))
    with col3:
        avg_dist = "N/A"  # You could calculate this if you have distance data
        st.metric("Avg Distance", avg_dist)
    
    # Display the map
    st.components.v1.html(enhanced_map._repr_html_(), height=600)
    
    # Add download option
    if st.button("💾 Download Map as HTML"):
        enhanced_map.save("enhanced_commute_map.html")
        st.success("Map saved as 'enhanced_commute_map.html'")
    
    return enhanced_map
