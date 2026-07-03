"""
Bayesian hierarchical model to predict expected costs for purchase orders

go to root and run:
    python -m src.model

"""

import pymc as pm
import arviz as az

def build_anomoly_model(df):
    
    vendors = df['Vendor_#'].astype('category')
    vendor_index = vendors.cat.codes.values
    num_vendors = len(vendors.cat.categories)
    
    with pm.Model() as model:
        
        # Hierarchial priors, learn the behavior of each vendor
        # the model learns to be 'lenient; with messy vendors', 'strict' with consistent vendors        vendor_index = df['Vendor_#'].astype('category').cat.codes.values

        # definie the liklihood of the cost
        vendor_mu = pm.Normal('vendor_mu', mu= df['Expected_Cost'].mean(), sigma = 100, shape=num_vendors)
        vendor_sigma = pm.HalfNormal('vendor_sigma', sigma = 50, shape=num_vendors)

        y_obs = pm.Normal('y_obs', mu=vendor_mu[vendor_index], sigma=vendor_sigma[vendor_index], observed=df['Actual_Cost'])

        # Inference
        trace = pm.sample(1000, return_inferencedata=True)

    return trace, vendors.cat.categories

def get_anomoly_scores(trace, df):
    # find prob that an actual cost is an outlier based on learned vendor paramaters
    # transforms 'trace' into a simple 0-100% anomoly alert
    
    #get learned means and std deviations for each vendor, store in 'trace' object
    mu_samples = trace.posterior['vendor_mu'].values.mean(axis=(0, 1))
    sigma_samples = trace.posterior['vendor_sigma'].values.mean(axis=(0, 1))

    #map the vendor_index to the specific learn mu and sigma
    vendor_indices = df['Vendor_#'].astype('category').cat.codes.values
    learned_mu = mu_samples[vendor_indices]
    learned_sigma = sigma_samples[vendor_indices]

    # calcualate the Z-Score: How many 'sigmas' aways from the mean is this PO?
    df['Anomaly_Score'] = (df['Actual_Cost'] - learned_mu).abs() / learned_sigma

    # If score is > 2 std deviations away, it's an anomaly
    df['Is_Anomaly'] = df['Anomaly_Score'] > 2.0

    return df


# generate audit_report
def generate_audit_report(df, silence_df):
    print("\n--- AUDIT REPORT ---")

    # 1. Price/Variance Anomalies
    ml_anomalies = df[df['Is_Anomaly'] == True]
    for _, row in ml_anomalies.iterrows():
        print(f"[!] PRICING ANOMALY: PO {row['PO_#']} | {row['Component_Type']} | Score: {row['Anomaly_Score']:.2f}")

    # 2. Billing Silence
    silence_anomalies = silence_df[silence_df['Is_Billing_Silence'] == True]
    for _, row in silence_anomalies.iterrows():
        print(f"[!] BILLING SILENCE: PO {row['PO_#']} | Open for {row['Days_Open']} days.")

if __name__== "__main__":
    import pandas as pd
    from src.pipeline import load_povar_data, triage_audit, find_billing_silence

    # 1. Load the Triage to filter data
    df = load_povar_data('data/raw_variances.csv')
    df = triage_audit(df)
    
    # 2. Build the model (Bayesian Analysis)
    trace, vendor_names = build_anomoly_model(df)
    df = get_anomoly_scores(trace, df)

    # 3. Statastics (Technical Sanity Check)
    print("\n--- MODEL PARAMTER SUMMARY ---")
    print(az.summary(trace))

    # 4. Comprehensive Audit Report
    silence_df = find_billing_silence(df)
    # generate_audit_report(df, silence_df)

    # 5. Final Actionable List (Focus)
    print("\n--- FINAL ACTIONABLE AUDIT LIST (High-Risk) ---")
    # to_review = df[df['Audit_Action'] == 'Review']
    to_review = df[(df['Audit_Action'] == 'Review') & (df['Is_Anomaly'] == True)]

    if not to_review.empty:
        # Displaying the POs that need manual human attention
        print(to_review[['PO_#', 'Vendor_#', 'Component_Type', 'Anomaly_Score', 'Is_Anomaly']])
    else:
        print("No high-risk anomalies found")
