# 🏠 HDB Resale Price Estimator (XGBoost + Explainability + Dashboard)

An end-to-end machine learning project that predicts **HDB resale prices in Singapore** using structured flat attributes such as **location, flat type, floor area, flat model, storey range, and remaining lease**.

This project includes:
- **Baseline Linear Regression / OLS** (econometrics-style interpretability)
- **XGBoost Regressor** (strong tabular ML performance)
- **SHAP explainability** (global + local feature contributions)
- **Streamlit dashboard** for interactive predictions + comparable flats (“comps”)

---

## ✨ Features

- Predict resale price from user-selected flat attributes  
- Show **empirical prediction interval** (based on test-set residuals)  
- Retrieve similar historical transactions as **comparable references (comps)**  
- Explain model behaviour using **SHAP summary plots**

---

## 📌 Problem Statement

Given a flat’s attributes, estimate its expected resale price and provide interpretable insights into which factors drive the prediction.

---

## 🧾 Dataset

Public dataset from **data.gov.sg** (HDB resale transactions).

> Housing & Development Board. (2021). *Resale flat prices based on registration date from Jan-2017 onwards (2026)* [Dataset]. data.gov.sg. Retrieved January 19, 2026 from  
https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view

---

## 🧠 Models

### 1) Baseline Model: Linear Regression / OLS
Used as a baseline for interpretability. Coefficients provide direction and magnitude of feature effects *(holding other variables constant)*.

### 2) Final Model: XGBoost Regressor
Chosen for strong performance on tabular data:
- Captures **non-linear patterns** and feature interactions  
- Works well with **one-hot encoded categorical features**  
- Supports explainability via **SHAP**

---

## 📊 Evaluation Metrics

Models are evaluated on a hold-out test set using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

Lower values indicate better predictive accuracy.

---

## 🔍 Explainability (SHAP)

SHAP is used to explain the model by showing:
- Which features matter most overall (**global importance**)  
- Why a specific flat prediction is higher/lower (**local explanation**)  

---

## 📈 Prediction Interval (Empirical)

The Streamlit app includes a prediction interval computed from the distribution of validation residuals.

⚠️ This is an **empirical prediction interval**, not a parametric statistical confidence interval.

---

## 🧩 Comparable Flats (“Comps”)

To make predictions more intuitive, the dashboard retrieves similar historical transactions based on:
- same **town / flat type**
- similar **floor area** (± tolerance)
- similar **remaining lease** (± tolerance)
- transactions within a **lookback window** (e.g., last 12 months)

This helps users compare model predictions against real resale prices.

---

## 🚀 Run Locally

### 1) Install dependencies
```bash
pip install -r requirements.txt
```
### 2) Start the Streamlit app
```bash
streamlit run app.py
```

---

## 🌐 Access the App (Live Demo)

You can try the deployed Streamlit app here:

🔗 https://hdb-resale-econometrics-othniel.streamlit.app/

---

## 📁 Project Structure
```bash
.
├── app.py                      # Streamlit dashboard
├── dataAnalysis.ipynb          # Training + evaluation notebook
├── resale_data.csv             # Dataset snapshot
├── hdb_xgb_pipeline.pkl        # Trained pipeline model
├── feature_cols.pkl            # Feature schema used during training
├── pi_bounds.pkl               # Empirical prediction interval bounds
├── requirements.txt
└── README.md
```
---
## ⚠️ Limitations

This model does not capture certain real-world pricing factors such as:

- renovation condition / interior quality
- exact unit facing and view
- proximity to MRT / amenities
- block-level micro-location effects

Therefore, some predictions may deviate from actual resale prices even for “similar” flats. Additionally, due to API retrieval limits, I have used historical data instead of implementing live data retrieval for the comparisons.

---

## ✅ Future Improvements

- Add geospatial features (e.g., MRT distance)
- Improve comps matching using KNN similarity search

---

## 👤 Author

Chang Tze Yi Othniel

Year 3 Business Analytics, National University of Singapore (NUS)
