"""
pipline.py
"""

import pandas as pd
import logging

# conf basic logging to track data intake
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_povar_data( file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)

        required_columns = [
            'Purchase_Order_#', 'Vendor_#', 'Expected_Cost', 
            'Actual_Cost', 'Variance_Amount'

        ]
    
        if not all( col in df.columns for col in required_columns ):
            raise ValueError( f"Data schema mistmatch. Expected columns: {required_columns}")
        
        initial_count = len(df)
        df = df.dropna(subset=['Expected_Cost', 'Actual_Cost'])

        if len(df) < initial_count:
            logger.warning( f"Dropped {intial_count - lef(df)}  rows due to missing data (cost vales)")

        logger.info(f"Successfully ingested {len(df)} records from {file_path}")
        return df


    except Exception as e:
        logger.error( "Failed to intake data: {e}") 
        raise

if __name__ == "__main__":
    try:
        data = load_povar_data('data/raw_variances.csv')
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")