"""
POVARA Bayesian Anomaly Detection Model

This module builds a hierarchical Bayesian model to estimate expected purchase

order costs by vendor and identify statistically unusual pricing behavior.

The workflow:

1. Loads purchase order variance data.

2. Applies audit triage rules from the pipeline.

3. Fits a vendor-level Bayesian model using PyMC.

4. Calculates anomaly scores and severity levels.

5. Generates an audit report and dashboard visualization.

Run from the project root:

    python -m src.model

Generated output:

    data/anomaly_report.png

"""

import os
import sys

# --- Windows Compatibility Safeguard ---
# PyTensor attempts to compile C++ code by default to accelerate Bayesian sampling.
# Because standard Windows environments lack the required 64-bit C++ toolchains, 
# this causes a fatal compilation crash. If the OS is Windows ("win32"), we set 
# the 'cxx' flag to empty, forcing PyTensor to bypass C-compilation and safely 
# fallback to the pure Python/NumPy backend.

if sys.platform == "win32":
    os.environ["PYTENSOR_FLAGS"] = "cxx="

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def build_anomaly_model(df):
    
    vendors = df['Vendor_#'].astype('category')
    vendor_index = vendors.cat.codes.values
    num_vendors = len(vendors.cat.categories)
    
    with pm.Model() as model:
        
        # Hierarchial priors, learn the behavior of each vendor
        # the model learns to be 'lenient; with messy vendors', 'strict' with consistent vendors       

        # define the likelihood of the cost
        vendor_mu = pm.Normal('vendor_mu', mu= df['Expected_Cost'].mean(), sigma = 100, shape=num_vendors)
        vendor_sigma = pm.HalfNormal('vendor_sigma', sigma = 50, shape=num_vendors)

        # LIKELIHOOD: Condition the model on our observed actual costs.
        y_obs = pm.Normal('y_obs', mu=vendor_mu[vendor_index], sigma=vendor_sigma[vendor_index], observed=df['Actual_Cost'])

        # INFERENCE: Generate posterior distributions via MCMC sampling.
        trace = pm.sample(1000, return_inferencedata=True, target_accept=0.9)

    return trace, vendors.cat.categories

def get_anomaly_scores(trace, df):
    # find prob that an actual cost is an outlier based on learned vendor paramaters
    # transforms 'trace' into a simple 0-100% anomaly alert
    
    #get learned means and std deviations for each vendor, store in 'trace' object
    mu_samples = trace.posterior['vendor_mu'].values.mean(axis=(0, 1))
    sigma_samples = trace.posterior['vendor_sigma'].values.mean(axis=(0, 1))

    #map the vendor_index to the specific learn mu and sigma
    vendor_indices = df['Vendor_#'].astype('category').cat.codes.values
    learned_mu = mu_samples[vendor_indices]
    learned_sigma = sigma_samples[vendor_indices]

    # calculate the Z-Score: How many 'sigmas' aways from the mean is this PO?
    df = df.copy()  
    df['Anomaly_Score'] = (df['Actual_Cost'] - learned_mu).abs() / learned_sigma

    # If score is > 2 std deviations away, it's an anomaly
    df['Is_Anomaly'] = df['Anomaly_Score'] > 2.0
    
    # calculate anomaly probability as percentage (more advanced than just the binary flag)
    # gives us a 0-100% confidence score that the cost is an outlier

    # Convert statistical rarity (p-value) into a human-readable confidence score
    tail_probability = (1 - norm.cdf(df['Anomaly_Score'])) * 2
    df['Anomaly_Confidence'] = (1 - tail_probability) * 100
    
    # add severity level for easier audit review
    def get_severity(score):
        if score > 3.5:
            return "CRITICAL"
        elif score > 2.5:
            return "HIGH"
        elif score > 2.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    df['Anomaly_Severity'] = df['Anomaly_Score'].apply(get_severity)

    return df

# generate audit_report
def generate_audit_report(df, silence_df):
    print("\n--- AUDIT REPORT ---")

    # 1. Price/Variance Anomalies
    ml_anomalies = df[df['Is_Anomaly'] == True].sort_values('Anomaly_Score', ascending=False)
    for _, row in ml_anomalies.iterrows():
        print(f"[!] PRICING ANOMALY: PO {row['PO_#']} | {row['Component_Type']} | "
              f"Score: {row['Anomaly_Score']:.2f} | Prob: {row['Anomaly_Confidence']:.1f}% | "
              f"Severity: {row['Anomaly_Severity']}")

    # 2. Billing Silence (If invoice 14+ days late)
    silence_anomalies = silence_df[silence_df['Is_Billing_Silence'] == True]
    for _, row in silence_anomalies.iterrows():
        print(f"[!] BILLING SILENCE: PO {row['PO_#']} | Open for {row['Days_Open']} days.")

def plot_anomalies(df):
    """Professional dashboard visualization including materiality threshold"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 1. Cost Distribution by Vendor
    sns.stripplot(data=df, x='Vendor_#', y='Actual_Cost', hue='Is_Anomaly',
                  palette={True: '#d32f2f', False: '#1976d2'}, 
                  jitter=True, alpha=0.8, s=7, ax=axes[0,0])
    axes[0,0].set_title('Actual Cost Distribution & Anomalies by Vendor')
    axes[0,0].set_ylabel('Actual Cost ($)')
    
    # 2. Variance with Materiality Threshold
    sns.scatterplot(data=df, x='Variance_Amount', y='Actual_Cost', 
                    hue='Is_Anomaly', style='Audit_Action',
                    palette={True: '#d32f2f', False: '#1976d2'}, 
                    alpha=0.8, s=60, ax=axes[0,1])
    axes[0,1].axvline(250, color='green', linestyle='--', alpha=0.7)
    axes[0,1].axvline(-250, color='green', linestyle='--', alpha=0.7)
    axes[0,1].set_title('Variance Amount vs Actual Cost\n(Green lines = ±$250 Materiality Threshold)')
    
    # 3. Anomaly Score Distribution
    sns.histplot(data=df, x='Anomaly_Score', hue='Is_Anomaly', 
                 bins=30, kde=True, ax=axes[1,0],
                 palette={True: '#d32f2f', False: '#1976d2'})
    axes[1,0].axvline(2.0, color='black', linestyle='--', label='Z-Score Threshold')
    axes[1,0].set_title('Distribution of Bayesian Anomaly Scores')
    axes[1,0].legend()
    
    # 4. Anomalies by Component Type
    anomaly_counts = df[df['Is_Anomaly']].groupby('Component_Type').size().sort_values(ascending=False)
    anomaly_counts.plot(kind='bar', ax=axes[1,1], color='#d32f2f')
    axes[1,1].set_title('Number of Anomalies by Component Type')
    axes[1,1].set_ylabel('Count')
    
    plt.suptitle('POVARA: Bayesian Anomaly Detection Dashboard', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('data/anomaly_report.png', dpi=300, bbox_inches='tight')
    print("[!] High-resolution dashboard saved to 'data/anomaly_report.png'")

if __name__== "__main__":
    import pandas as pd
    from src.pipeline import load_povar_data, triage_audit, find_billing_silence

    # 1. Load the Triage to filter data
    df = load_povar_data('data/expanded_variances.csv')
    df = triage_audit(df)
    
    # 2. Build the model (Bayesian Analysis)
    trace, vendor_names = build_anomaly_model(df)
    df = get_anomaly_scores(trace, df)

    plot_anomalies(df)

    # 3. Statastics (Technical Sanity Check)
    print("\n--- MODEL PARAMATER SUMMARY ---")
    print(az.summary(trace))

    # 4. Comprehensive Audit Report
    silence_df = find_billing_silence(df)
    generate_audit_report(df, silence_df)

    # 5. Final Actionable List (Focus)
    print("\n--- FINAL ACTIONABLE AUDIT LIST (High-Risk) ---")
    to_review = df[(df['Audit_Action'] == 'Review') & (df['Is_Anomaly'] == True)]

    if not to_review.empty:
        print(to_review[['PO_#', 'Vendor_#', 'Component_Type', 
                         'Anomaly_Score', 'Anomaly_Confidence', 
                         'Anomaly_Severity', 'Is_Anomaly']])
    else:
        print("No high-risk anomalies found")