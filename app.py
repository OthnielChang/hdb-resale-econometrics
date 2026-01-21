import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="HDB Resale Price Estimator", layout="centered")

st.title("HDB Resale Price Estimator")
st.caption("Predict resale price from flat attributes using XGBoost")

# Load model & columns
model = joblib.load("hdb_xgb_pipeline.pkl")
feature_cols = joblib.load("feature_cols.pkl")

df = pd.read_csv("resale_data.csv")

towns = sorted(df["town"].dropna().unique().tolist())
flat_types = sorted(df["flat_type"].dropna().unique().tolist())
storey_ranges = sorted(df["storey_range"].dropna().unique().tolist())
flat_models = sorted(df["flat_model"].dropna().unique().tolist())

town = st.selectbox("Town", towns)
flat_type = st.selectbox("Flat Type", flat_types)
storey_range = st.selectbox("Storey Range", storey_ranges)
flat_model = st.selectbox("Flat Model", flat_models)

floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=20.0, max_value=250.0, value=90.0, step=1.0)

remaining_lease_years = st.number_input("Remaining Lease (years)", min_value=0.0, max_value=99.0, value=60.0, step=0.5)

# Prediction
if st.button("Predict Price"):
    X_input = pd.DataFrame([{
        "town": town,
        "flat_type": flat_type,
        "flat_model": flat_model,
        "storey_range": storey_range,
        "floor_area_sqm": floor_area_sqm,
        "remaining_lease_years": remaining_lease_years,
    }])

    X_input = X_input[feature_cols]

    # Model predicts log_price
    pred_log = float(model.predict(X_input)[0])
    pred_price = float(np.exp(pred_log))

    st.subheader("Prediction")
    st.metric("Predicted Resale Price", f"${pred_price:,.0f}")

    mae_log = 0.08536281436681747
    low = np.exp(pred_log - mae_log)
    high = np.exp(pred_log + mae_log)

    st.caption("Approximate range based on typical model error (MAE).")
    st.write(f"Likely range: **${low:,.0f} – ${high:,.0f}**")
