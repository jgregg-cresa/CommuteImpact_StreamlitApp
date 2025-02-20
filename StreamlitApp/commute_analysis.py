import pandas as pd
import os
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

@dataclass
class CommuteData:
    """Data class to store commute analysis results"""
    dataframes: List[pd.DataFrame]
    dataframes_dict: Dict[str, pd.DataFrame]
    split_files: List[str]

class CommuteAnalyzer:
    """Class to handle commute data analysis and transformation"""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.commute_data = self._read_commute_results()
        
    def _read_commute_results(self) -> CommuteData:
        """Read commute result files from specified folder"""
        dfs = []
        split_f = []
        dicty = {}
        
        for file in os.listdir(self.folder_path):
            if not os.path.isfile(os.path.join(self.folder_path, file)):
                continue
                
            str_split = file.split("_")[-1].split(".csv")
            split_f.append(str_split)
            
            file_path = os.path.join(self.folder_path, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
            dicty[str_split[0]] = df
            
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
            return -10
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
            
            # Process method-specific dataframe
            popped_col = df.pop('Method')
            dist_pos = df.columns.get_loc('Zipcode_dist')
            df_cleaned = df.iloc[:, :dist_pos]
            
            time_pos = df_cleaned.columns.get_loc('Employee_ID') + 1
            df_cleaned.rename(columns={df_cleaned.columns[time_pos]: 'CurrentCommute_Time'}, inplace=True)
            
            # Rename potential location columns
            potential_cols = df_cleaned.columns[time_pos + 1:]
            for idx, col in enumerate(potential_cols, 1):
                df_cleaned.rename(columns={col: f'Potential_Location_{idx}'}, inplace=True)
                
            df_cleaned.insert(1, 'Method', popped_col)
            cleaned_dfs.append(df_cleaned)
        
        # Merge dataframes if multiple methods exist
        if len(methods) > 1:
            final_df = pd.concat(cleaned_dfs).reset_index(drop=True)
        else:
            final_df = cleaned_dfs[0]
            
        return final_df
    
    def transform_for_visualization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data for visualization"""
        # Get column positions
        id_cols = ['Employee_ID', 'Method', 'Geoid', 'Zipcode']
        value_cols = [col for col in df.columns if col not in id_cols]
        
        # Melt dataframe
        melted_df = pd.melt(df, id_vars=id_cols, value_vars=value_cols)
        
        # Add time buckets and changes
        melted_df['Commute_Time_Bucket'] = melted_df['value'].apply(self._commute_time_bucket)
        
        # Calculate time changes for potential locations
        base_times = melted_df[melted_df['variable'] == 'CurrentCommute_Time']['value'].values
        potential_mask = melted_df['variable'].str.startswith('Potential_Location')
        
        melted_df.loc[potential_mask, 'Time_Change'] = (
            melted_df.loc[potential_mask, 'value'] - 
            melted_df.loc[potential_mask, 'value'].iloc[0]
        )
        
        # Add time change categories
        melted_df.loc[potential_mask, 'Time_Change_Bucket'] = (
            melted_df.loc[potential_mask, 'Time_Change'].apply(self._calculate_time_change)
        )
        melted_df.loc[potential_mask, 'Time_Change_Category'] = (
            melted_df.loc[potential_mask, 'Time_Change'].apply(self._determine_time_change)
        )
        
        return melted_df

def main(folder_path: str, output_path: str = None) -> pd.DataFrame:
    """Main function to process commute analysis data
    
    Args:
        folder_path: Path to folder containing commute CSV files
        output_path: Optional path to save processed data
        
    Returns:
        Processed DataFrame ready for visualization
    """
    analyzer = CommuteAnalyzer(folder_path)
    
    # Process the data
    merged_df = analyzer.process_commute_data()
    final_df = analyzer.transform_for_visualization(merged_df)
    
    # Save to CSV if output path provided
    if output_path:
        final_df.to_csv(output_path, index=False)
    
    return final_df

if __name__ == "__main__":
    # Example usage
    folder_path = "./commute_data"
    output_path = "./processed_commute_data.csv"
    
    processed_data = main(folder_path, output_path)
    print("Data processing complete!")