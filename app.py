import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(page_title="HDB Resale Price Estimator", layout="centered")

st.title("HDB Resale Price Estimator")
st.caption("Predict resale price from flat attributes using XGBoost")

# Load model & columns
model = joblib.load("hdb_xgb_pipeline.pkl")
feature_cols = joblib.load("feature_cols.pkl")

df = pd.read_csv("resale_data.csv")

def lease_to_years(s):
    if pd.isna(s):
        return np.nan
    
    s = str(s).lower()
    
    years = re.search(r"(\d+)\s*year", s)
    months = re.search(r"(\d+)\s*month", s)
    
    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0
    
    return y + m/12

df["remaining_lease_years"] = df["remaining_lease"].apply(lease_to_years)
df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")

towns = sorted(df["town"].dropna().unique().tolist())
flat_types = sorted(df["flat_type"].dropna().unique().tolist())
storey_ranges = sorted(df["storey_range"].dropna().unique().tolist())
flat_models = sorted(df["flat_model"].dropna().unique().tolist())

town = st.selectbox("Town", towns)
flat_type = st.selectbox("Flat Type", flat_types)
storey_range = st.selectbox("Storey Range", storey_ranges)
flat_model = st.selectbox("Flat Model", flat_models)

floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=20.0, max_value=250.0, value=90.0, step=1.0)

remaining_lease_years = st.number_input("Remaining Lease (years)", min_value=0.0, max_value=99.0, value=60.0, step=1.0)

st.subheader("Comparable Flats Settings")
area_tol = st.slider("Comparable area tolerance (± sqm)", 5, 30, 10)
lease_tol = st.slider("Comparable lease tolerance (± years)", 1, 20, 5)
months_back = st.slider("Lookback period (months)", 3, 36, 12)

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

    low_q, high_q = joblib.load("pi_bounds.pkl")

    pred_log = float(model.predict(X_input)[0])

    pred_price = float(np.exp(pred_log))
    lower_price = float(np.exp(pred_log + low_q))
    upper_price = float(np.exp(pred_log + high_q))

    st.metric("Predicted Resale Price", f"${pred_price:,.0f}")
    st.write(f"**90% Prediction Interval:** ${lower_price:,.0f} – ${upper_price:,.0f}")
    st.caption("Interval is estimated empirically from test-set residuals.")

    X_input["pred_price"] = pred_price

    cutoff = df["month"].max() - pd.DateOffset(months=months_back)

    comps = df[
        (df["month"] >= cutoff) &
        (df["town"] == town) &
        (df["flat_type"] == flat_type) &
        (df["flat_model"] == flat_model) &
        (df["storey_range"] == storey_range) &
        (df["floor_area_sqm"].between(floor_area_sqm - area_tol, floor_area_sqm + area_tol)) &
        (df["remaining_lease_years"].between(remaining_lease_years - lease_tol, remaining_lease_years + lease_tol))
    ].copy()

    st.subheader("Comparable Flats (Actual Transactions)")
    st.dataframe(comps[["month","town","flat_type","flat_model","storey_range","floor_area_sqm","remaining_lease_years","resale_price"]].head(20))


