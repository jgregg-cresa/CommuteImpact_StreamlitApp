# utils/data_processor.py
import streamlit as st
import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

    def filter_by_commute_time(self, df: pd.DataFrame, max_commute_time: int) -> Tuple[pd.DataFrame, int, int]:
        """
        Filters the DataFrame to include only employees with a current commute time
        less than or equal to the specified maximum.
        """
        # Ensure 'value' column is numeric for filtering
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Get all unique employee IDs before filtering
        all_employees = df['Employee_ID'].unique()
        total_employees = len(all_employees)
        
        # Identify current commute times
        current_commute_times = df[df['variable'] == 'CurrentCommute_Time']
        
        # Find employees whose current commute is within the max time
        valid_employees = current_commute_times[
            current_commute_times['value'] <= max_commute_time
        ]['Employee_ID'].unique()
        
        # Filter the full DataFrame to only include data for valid employees
        filtered_df = df[df['Employee_ID'].isin(valid_employees)].copy()
        
        remaining_employees = len(valid_employees)

        return filtered_df, total_employees, remaining_employees
    
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
            # Only insert Method column if it doesn't already exist
            if 'Method' not in df.columns:
                df.insert(1, 'Method', method)
            cleaned_dfs.append(df)
        
        return pd.concat(cleaned_dfs).reset_index(drop=True)
    
    def transform_for_visualization(self, df: pd.DataFrame, destinations_df: pd.DataFrame) -> pd.DataFrame:
        """Transform data for visualization in the target format"""
        # Identify duration columns
        duration_cols = [col for col in df.columns if col.startswith('Duration')]
        base_cols = ['Employee_Number', 'Method', 'Zipcode', 'Latitude', 'Longitude', 'ADDRESS_FULL']
        
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
                    'value': str(self._commute_time_bucket(commute_time)),
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
        # Sort by a single column
        result_df = result_df[['Employee_Number', 'Method', 'Latitude', 'Longitude', 'Zipcode', 'variable', 'value', 'names']]
        # Sort by a single column
        result_df = result_df.sort_values(by='variable')
        
        return result_df
        
    def filter_by_commute_time(self, df: pd.DataFrame, max_commute_minutes: float = 120.0) -> tuple:
        """Filter employees based on current commute time"""
        # Get unique employee IDs
        all_employees = df['Employee_Number'].unique()
        
        # Get employees with current commute time > max_commute_minutes
        # Only look at rows where variable is 'CurrentCommute_Time' and make sure to handle conversion errors
        commute_time_df = df[df['variable'] == 'CurrentCommute_Time'].copy()
        
        # Safely convert values to float, handling any non-numeric values
        try:
            commute_time_df['numeric_value'] = pd.to_numeric(commute_time_df['value'], errors='coerce')
            employees_to_filter = commute_time_df[
                commute_time_df['numeric_value'] > max_commute_minutes
            ]['Employee_Number'].unique()
        except Exception as e:
            st.warning(f"Error converting commute times: {e}. Using all employees.")
            employees_to_filter = []
        
        # Filter out these employees from the DataFrame
        filtered_df = df[~df['Employee_Number'].isin(employees_to_filter)]
        
        return filtered_df, len(all_employees), len(all_employees) - len(employees_to_filter)

def create_simplified_dashboard(filtered_df, destinations_df):
    """
    Create a simplified dashboard showing commute time distributions for all locations
    """
    
    # Get unique destinations for analysis
    destinations = destinations_df['ADDRESS_FULL'].tolist()
    
    st.header("Commute Time Analysis - All Locations")
    
    # Get current location data
    current_data = filtered_df[
        (filtered_df['variable'] == 'Current_Commute_Time_Bucket') & 
        (filtered_df['names'] == destinations[0])
    ].copy()
    
    if len(current_data) == 0:
        st.error("No current commute data found")
        return
    
    # Prepare data for the chart
    time_buckets = ['0-15 Minutes', '15-30 Minutes', '30-45 Minutes', '45-60 Minutes', '60 Minutes +']
    chart_data = {}
    
    # Current location data
    current_location_name = destinations[0] if ',' in destinations else destinations
    current_buckets = current_data['value'].value_counts().reindex(time_buckets, fill_value=0)
    chart_data[f"Current: {current_location_name}"] = current_buckets.values
    
    # Get all potential locations data
    potential_locations = []
    # for dest_idx, destination in enumerate(destinations[1:], 1):
    #     potential_data = filtered_df[
    #         (filtered_df['variable'] == f'Potential_Commute_Time_Reduced_Bucket_{dest_idx}') & 
    #         (filtered_df['names'] == destination)
    #     ].copy()
        
    #     if len(potential_data) > 0:
    #         potential_location_name = destination.split(',')[0] if ',' in destination else destination
    #         potential_buckets = potential_data['value'].value_counts().reindex(time_buckets, fill_value=0)
    #         chart_data[f"Potential: {potential_location_name}"] = potential_buckets.values
    #         potential_locations.append((dest_idx, destination, potential_location_name))

    for dest_idx, destination in enumerate(destinations[1:], 1):
        potential_data = filtered_df[
            (filtered_df['variable'] == f'Potential_Commute_Time_Reduced_Bucket_{dest_idx}') & 
            (filtered_df['names'] == destination)
        ].copy()
    
        potential_location_name = destination.split(',') if ',' in destination else destination
        
        if len(potential_data) > 0:
            potential_buckets = potential_data['value'].value_counts().reindex(time_buckets, fill_value=0)
        else:
            # Ensure the location still shows up in the chart with zero values
            potential_buckets = pd.Series([0] * len(time_buckets), index=time_buckets)
    
        chart_data[f"Potential: {potential_location_name}"] = potential_buckets.values
        potential_locations.append((dest_idx, destination, potential_location_name))
    
    # Create the dynamic chart
    fig = go.Figure()
    
    # Color palette for different locations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Add bars for each location
    for idx, (location_name, values) in enumerate(chart_data.items()):
        fig.add_trace(go.Bar(
            x=['0-15 Min', '15-30 Min', '30-45 Min', '45-60 Min', '60+ Min'],
            y=values,
            name=location_name,
            marker_color=colors[idx % len(colors)],
            text=values,
            textposition='auto',
        ))
    
    # Update layout
    fig.update_layout(
        title="Employees by Commute Time - 15 Minute Intervals (All Locations)",
        xaxis_title="Commute Time Range",
        yaxis_title="Number of Employees",
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.write("DEBUG - Chart Data Keys:", list(chart_data.keys()))

    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics table
    st.subheader("Location Comparison Summary")
    
    summary_data = []
    
    # Current location metrics
    current_commute_data = filtered_df[
        (filtered_df['variable'] == 'CurrentCommute_Time') & 
        (filtered_df['names'] == destinations[0])
    ]
    current_times = pd.to_numeric(current_commute_data['value'], errors='coerce').dropna()
    
    summary_data.append({
        'Location': f"Current: {current_location_name}",
        'Total Employees': len(current_times),
        'Average Commute (min)': f"{current_times.mean():.1f}",
        'Employees ≤15 min': int(current_buckets['0-15 Minutes']),
        'Employees ≤30 min': int(current_buckets['0-15 Minutes'] + current_buckets['15-30 Minutes']),
        'Employees >60 min': int(current_buckets['60 Minutes +'])
    })
    
    # Potential locations metrics
    for dest_idx, destination, potential_location_name in potential_locations:
        potential_commute_data = filtered_df[
            (filtered_df['variable'] == f'Potential_Location_{dest_idx}') & 
            (filtered_df['names'] == destination)
        ]
        
        change_data = filtered_df[
            (filtered_df['variable'] == f'Change_Commute_{dest_idx}') & 
            (filtered_df['names'] == destination)
        ]
        
        time_category_data = filtered_df[
            (filtered_df['variable'] == f'Commute_Time_Category_Bucket_{dest_idx}') & 
            (filtered_df['names'] == destination)
        ]
        
        potential_times = pd.to_numeric(potential_commute_data['value'], errors='coerce').dropna()
        time_changes = pd.to_numeric(change_data['value'], errors='coerce').dropna()
        improved_employees = len(time_category_data[time_category_data['value'] == 'Time Reduced'])
        
        potential_buckets = chart_data[f"Potential: {potential_location_name}"]
        
        summary_data.append({
            'Location': f"Potential: {potential_location_name}",
            'Total Employees': len(potential_times),
            'Average Commute (min)': f"{potential_times.mean():.1f}",
            'Employees ≤15 min': int(potential_buckets[0]),
            'Employees ≤30 min': int(potential_buckets[0] + potential_buckets[1]),
            'Employees >60 min': int(potential_buckets[4]),
            'Avg Impact (min)': f"{time_changes.mean():.1f}",
            'Employees Improved': improved_employees
        })
    
    # Display summary table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    st.divider()

# Original Streamlit App Functions (slightly modified)
def process_origins(df):
    """Process the origins DataFrame"""
    # Ensure Employee_Number exists
    if 'Employee_Number' not in df.columns:
        df.insert(0, 'Employee_Number', range(1, len(df) + 1))
    
    # Handle geoid column
    if 'geoid' not in df.columns:
        df.insert(1, 'geoid', range(1, len(df) + 1))
    
    # Handle employee expansion
    if 'count_employees' in df.columns:
        df = df.loc[df.index.repeat(df['count_employees'])].reset_index(drop=True)
        df['Employee_Number'] = range(1, len(df) + 1)
    
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
