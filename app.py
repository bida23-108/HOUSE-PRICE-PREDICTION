
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Prediction App")

st.markdown("""
This app predicts the **median house value** based on California housing data features.
Fill in the inputs below to get an estimate.
""")

# User input sliders
median_income = st.slider("Median Income", 0.0, 20.0, 5.0)
total_rooms = st.slider("Total Rooms", 0, 10000, 2000)
total_bedrooms = st.slider("Total Bedrooms", 0, 2000, 400)
population = st.slider("Population", 0, 5000, 1000)
households = st.slider("Households", 0, 3000, 800)
latitude = st.slider("Latitude", 32.0, 42.0, 35.0)
longitude = st.slider("Longitude", -125.0, -114.0, -120.0)
housing_median_age = st.slider("Housing Median Age", 1, 52, 20)

# Ocean proximity radio button
st.markdown("#### Ocean Proximity (select one)")
ocean_proximity = {
    "<1H OCEAN": 0,
    "INLAND": 0,
    "ISLAND": 0,
    "NEAR BAY": 0,
    "NEAR OCEAN": 0
}
selected_ocean = st.radio("Location", list(ocean_proximity.keys()))
ocean_proximity[selected_ocean] = 1

# Create DataFrame for prediction
input_data = pd.DataFrame([{**{
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income
}, **{
    f"ocean_proximity_{key}": value for key, value in ocean_proximity.items()
}}])

# Ensure all expected columns exist and are in correct order
expected_columns = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN'
]

for col in expected_columns:
    if col not in input_data:
        input_data[col] = 0
input_data = input_data[expected_columns]

# Scale and predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

st.subheader(f"💰 Predicted Median House Value: **${int(prediction[0]):,}**")
