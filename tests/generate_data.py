"""
generate synthetic data for testing
"""

import pandas as pd
import numpy as np

def generate_test_data(n=100):
    vendors = ['V101', 'V102', 'V103', 'V104']
    data = {
        'PO_#': [f'PO{1000+i}' for i in range(n)],
        'Vendor_#': np.random.choice(vendors, n),
        'Expected_Cost': np.random.uniform(100, 2000, n),
        'Actual_Cost': np.random.uniform(100, 2000, n),
        'Receipt_Date': pd.to_datetime('2026-06-01') + pd.to_timedelta(np.random.randint(0, 30, n), unit='D')
    }
    # Randomly introduce missing invoices
    df = pd.DataFrame(data)
    df.loc[::10, 'Invoice_Date'] = np.nan # Simulate silence
    df.to_csv('data/expanded_variances.csv', index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate new test data")
    args = parser.parse_args()
    
    if args.generate:
        generate_test_data(n=200)
        print("Test data generated successfully.")