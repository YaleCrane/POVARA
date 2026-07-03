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

def triage_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies filters to data before sending to the model for ML analysis
    """

    df['Audit_Action'] = 'Review'

    # If variance is small, mark as AUTO_RECONCILED
    mask = df['Variance_Amount'].abs() < 250
    df.loc[mask, 'Audit_Action'] = 'AUTO_RECONCILED'

    logger.info(f"Triage complete: {len(df[df['Audit_Action'] == 'AUTO_RECONCILED'])} lines auto-reconciled")
    
    return df

def find_billing_silence(df: pd.DataFrame, threshold_days: int = 14) -> pd.DataFrame:
    """
    Flags items that have not been invoided within the vendors average lag time.
    """

    # find line where Invoice_Date is missing
    open_lines = df[df['Invoice_Date'].isna()].copy()

    # calculate days since receipt
    today = pd.Timestamp('2026-07-03')
    open_lines['Days_Open'] = (today - open_lines['Receipt_Date']).dt.days

    # Flag lines open longer than threshold
    open_lines["Is_Billing_Silence"] = open_lines['Days_Open'] > threshold_days

    return open_lines




if __name__ == "__main__":
    # Test block to verify pipeline
    try:
        data = load_povar_data('data/raw_variances.csv')
        print(data.head())
    except Exception as e:
        print(f"Test failed: {e}")