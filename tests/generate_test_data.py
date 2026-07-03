"""
generate synthetic data for testing

run in root:
python -m tests.generate_test_data --generate

"""

import pandas as pd
import numpy as np
import os

def generate_test_data(n=100):

    # imported OS to be able to send to your target
    # send test csv here:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)

    vendors = ['V101', 'V102', 'V103', 'V104']
    components = ['Raw Material', 'Freight', 'Demurrage', 'Bundled_Material_Freight']
    
    data = {
        'PO_#': [f'PO{1000+i}' for i in range(n)],
        'Vendor_#': np.random.choice(vendors, n),
        'Component_Type': np.random.choice(components, n),
        'Expected_Cost': np.round(np.random.uniform(100, 2000, n),2),
        'Actual_Cost': np.round(np.random.uniform(100, 2000, n),2),
        'Receipt_Date': pd.to_datetime('2026-06-01') + pd.to_timedelta(np.random.randint(0, 30, n), unit='D')
    }
    # Randomly introduce missing invoices
    df = pd.DataFrame(data)
    df.loc[::10, 'Invoice_Date'] = np.nan # Simulate silence

    # send to explicit path
    output_path = os.path.join(output_dir, 'expanded_variances.csv')
    df.to_csv(output_path, index=False)
    print(f"Test data generated successfully as: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate new test data")
    args = parser.parse_args()
    
    if args.generate:
        generate_test_data(n=200)
        print("Test data generated successfully.")