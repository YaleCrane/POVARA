"""
go to root and run:
    python -m src.pipeline.py
"""

import pandas as pd
import logging

# Configure logging for audit trails
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_povar_data(file_path: str) -> pd.DataFrame:
    """
    Ingests PO component data, validates the schema, and engineers 
    features necessary for Bayesian anomaly detection.
    """
    try:
        # Load data, forcing date parsing
        df = pd.read_csv(file_path, parse_dates=['Receipt_Date', 'Invoice_Date'])
        
        # Schema validation: We require the core transactional DNA for POVAR review
        required_columns = [
            'PO_#', 'Vendor_#', 'Component_Type', 
            'Expected_Cost', 'Actual_Cost', 'Receipt_Date'
        ]
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Schema mismatch. Missing required columns.")
            
        # Basic data integrity: drop rows missing core cost/receipt identifiers
        initial_count = len(df)
        df = df.dropna(subset=['Expected_Cost', 'Actual_Cost', 'Receipt_Date'])
        
        # 1. Invoice_Lag_Days: Helps model distinguish between 'pending' and 'delayed'
        df['Invoice_Lag_Days'] = (df['Invoice_Date'] - df['Receipt_Date']).dt.days
        
        # 2. Variance_Amount: Accounting standard (Expected - Actual)
        df['Variance_Amount'] = df['Expected_Cost'] - df['Actual_Cost']
        
        # 3. Is_Expected_Zero: Feature for detecting 'Ghost' costs like Demurrage
        df['Is_Expected_Zero'] = df['Expected_Cost'] == 0
        
        logger.info(f"Successfully processed {len(df)} component records. Features engineered.")
        return df

    except Exception as e:
        logger.error(f"Pipeline failure: {e}")
        raise

if __name__ == "__main__":
    # Test block to verify pipeline
    try:
        data = load_povar_data('data/raw_variances.csv')
        print(data.head())
    except Exception as e:
        print(f"Test failed: {e}")