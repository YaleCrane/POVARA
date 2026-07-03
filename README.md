# POVAR Sentinel: Automated Purchase Order Variance Audit & Anomaly Detection

![POVAR Dashboard](data/anomaly_report.png)

**POVAR Sentinel** is a hybrid audit automation system designed to streamline the reconciliation of high-volume procurement transactions. It intelligently combines rule-based financial controls with probabilistic Bayesian modeling to detect pricing anomalies, auto-reconcile low-risk items, and flag operational bottlenecks.

By reducing manual review workload while maintaining strong audit standards, this system demonstrates a practical application of data science in financial controls and procurement operations.

## Key Features

- **Materiality-Based Triage**: Automatically reconciles low-variance transactions (**±$250**) to eliminate noise and focus auditor attention on high-risk items.
- **Hierarchical Bayesian Anomaly Detection**: Built with PyMC to learn vendor-specific pricing behaviors and flag statistically significant outliers (Z-score > 2.0).
- **Billing Silence Monitoring**: Identifies stalled or missing invoices with configurable operational thresholds.
- **Data Quality Gate**: Enforces strict schema validation and comprehensive audit logging.
- **Actionable Insights**: Generates prioritized reports with severity scoring and rich visualizations.

## Tech Stack

- **Bayesian Modeling**: PyMC, ArviZ
- **Data Pipeline**: Pandas, NumPy
- **Visualization**: Seaborn, Matplotlib, Plotly
- **Other**: Python, logging, synthetic data generation

## Project Highlights

- Developed a **hierarchical Bayesian model** that adapts to individual vendor pricing variability.
- Designed an end-to-end production-style pipeline with automated triage and reporting.
- Created a multi-panel analytical dashboard that visualizes both statistical anomalies and business materiality rules.
- Built a flexible synthetic data generator for testing and demonstration.

## Dashboard Preview

The visualization below showcases:
- Cost distributions and detected anomalies by vendor
- Variance analysis with clear **±$250 materiality thresholds**
- Anomaly score distribution
- Breakdown by component type

![POVAR Analytics Dashboard](data/anomaly_report.png)

## Project Structure

```text
POVAR/
├── data/                    # Datasets and generated visualizations
├── src/                     # Core pipeline, Bayesian model, and reporting
├── tests/                   # Synthetic data generator and validation
├── requirements.txt
└── README.md