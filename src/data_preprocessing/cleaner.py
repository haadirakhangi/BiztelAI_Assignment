import pandas as pd
import numpy as np

class DataCleaner:
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame. 
        Categorical columns will be filled with 'unknown',
        Numerical columns will be filled with the median, since more robust than mean.
        """
        print("Handling missing values...")
        for col in df.columns:
            if df[col].isnull().any():
                print(f"Column '{col}' has {df[col].isnull().sum()} missing values.")
                if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                    df[col].fillna('Unknown', inplace=True)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                print(f"Column {col} has no missing values")
        return df
    
    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate records."""
        print(f"Handling duplicate values...\nInitial rows: {len(df)}")
        df_deduplicated = df.drop_duplicates()
        print(f"\nRows after dropping exact duplicates: {len(df_deduplicated)}")
        return df_deduplicated
    
    def correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures columns have appropriate data types."""
        print("Correcting data types...")
        if 'agent' in df.columns:
            df['agent'] = pd.Categorical(df['agent'])
        if 'sentiment' in df.columns:
            df['sentiment'] = pd.Categorical(df['sentiment'])
        if 'turn_rating' in df.columns:
            df['turn_rating'] = pd.Categorical(df['turn_rating'])
        if 'config' in df.columns:
            df['config'] = pd.Categorical(df['config'])
        return df
    
    def clean_data(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by handling missing values, duplicates, and correcting data types.
        Returns a cleaned DataFrame.
        """
        df = self.handle_missing_values(df)
        df = self.handle_duplicates(df)
        df = self.correct_data_types(df)
        print("Data cleaning process completed.")
        return df