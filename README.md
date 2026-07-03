# POVAR Sentinel: Automated Audit & Anomaly Detection

## Overview
POVAR (Purchase Order Variance Audit & Reconciliation) is a hybrid automated audit system designed to streamline the reconciliation of high-volume procurement transactional data. By combining rule-based financial controls with probabilistic machine learning, the system automates triage and flags systemic pricing risks, significantly reducing manual auditor intervention.



## Core Features
* **Materiality-Based Triage:** Automatically reconciles low-variance transactions (< $250) to filter "noise" and prioritize high-risk items.
* **Bayesian Anomaly Detection:** Utilizes a hierarchical Bayesian model (built with `PyMC`) to learn vendor-specific pricing behaviors and flag statistically significant cost outliers (Z-score > 2.0).
* **Operational Sentinel:** Monitors for "Billing Silence"—a custom logic gate that identifies stalled or missing invoices, flagging potential process bottlenecks.
* **Data Quality Gate:** Enforces a rigid schema to ensure only high-integrity data enters the analytic pipeline.

## Technical Stack
* **Modeling:** Python, PyMC, ArviZ (Bayesian inference).
* **Data Processing:** Pandas, NumPy (Data cleaning, feature engineering).
* **Audit Logic:** Custom triage algorithms based on federal excise tax and inventory variance standards.
* **Development:** Modular, branching architecture designed for production-grade audit trails.

## Project Structure
```text
POVAR/
├── data/               # Transactional datasets
├── src/                # Core pipeline, model architecture, and reporting
├── tests/              # Synthetic data generation and validation scripts
└── .venv/              # Environment dependencies POVAR variance finder
