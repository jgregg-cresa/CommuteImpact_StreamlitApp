# utils/data_processor.py

import pandas as pd
import re
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
