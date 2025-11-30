# enhanced Streamlit UI for car price prediction
# includes Car_Name dropdown, sliders, better layout, and feature importance (for linear model)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

MODEL_FILE = "car_price_model.pkl"
SCALER_FILE = "scaler.pkl"
COLS_FILE = "feature_columns.pkl"
CSV_FILE = "car data.csv"

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Enhanced Car Selling Price Predictor")
st.write("Improved UI with dropdowns, sliders, and feature importance.")

# -------------------------------
# Load artifacts
# -------------------------------

def load_artifacts():
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    with open(COLS_FILE, 'rb') as f:
        cols = pickle.load(f)
    return model, scaler, cols

model, scaler, feature_cols = load_artifacts()

# -------------------------------
# Load CSV for Car_Name options
# -------------------------------
if os.path.exists(CSV_FILE):
    df_names = pd.read_csv(CSV_FILE)
    car_names = sorted(df_names['Car_Name'].unique())
else:
    car_names = []

# -------------------------------
# Sidebar for Inputs
# -------------------------------
st.sidebar.header("Input Car Details")

car_name = st.sidebar.selectbox("Car Name", car_names if car_names else ["Not Available"])

year = st.sidebar.slider("Year", min_value=1990, max_value=2025, value=2015)
kms_driven = st.sidebar.slider("Kms Driven", min_value=0, max_value=300000, value=30000, step=500)
present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0, step=0.1)
owner = st.sidebar.selectbox("Owner Count", [0, 1, 2, 3])

fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# -------------------------------
# Prediction Logic
# -------------------------------
def predict_price():
    row = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
    
    # numeric
    if 'Year' in row.columns: row.at[0, 'Year'] = year
    if 'Kms_Driven' in row.columns: row.at[0, 'Kms_Driven'] = kms_driven
    if 'Present_Price' in row.columns: row.at[0, 'Present_Price'] = present_price
    if 'Owner' in row.columns: row.at[0, 'Owner'] = owner

    # fuel
    fmap = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    if 'Fuel_Type' in row.columns: row.at[0, 'Fuel_Type'] = fmap[fuel_type]

    # seller\_dealer
    if 'Seller_Type_Dealer' in row.columns:
        row.at[0, 'Seller_Type_Dealer'] = 1 if seller_type == 'Dealer' else 0

    # transmission_manual
    if 'Transmission_Manual' in row.columns:
        row.at[0, 'Transmission_Manual'] = 1 if transmission == 'Manual' else 0

    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    return pred, row

# -------------------------------
# Main Content
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict Selling Price"):
        predicted_price, input_row = predict_price()
        st.success(f"### Estimated Selling Price: **₹ {predicted_price:.2f} lakhs**")
        st.write("#### Input Feature Vector")
        st.dataframe(input_row)

with col2:
    st.write("### Feature Importance (Linear Model Coefficients)")
    try:
        coeff_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', ascending=False)
        st.bar_chart(coeff_df.set_index('Feature'))
    except:
        st.warning("Feature importance not available for this model.")

st.markdown("---")
st.caption("Enhanced Streamlit UI — Car Dropdown, Sliders, Better Layout, Feature Coefficients.")
