import joblib
import numpy as np
import streamlit as st

# Load trained model and label encoder
model = joblib.load("xgboost_crop_recommendation.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("🌾 Crop Recommendation System")
st.markdown("### Enter Soil and Environmental Conditions to Get the Best Crop Recommendation")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50, step=1)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50, step=1)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50, step=1)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=0.1)

# Predict button
if st.button("Recommend Crop 🌱"):
    # Prepare input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Make prediction
    prediction = model.predict(input_data)
    recommended_crop = label_encoder.inverse_transform(prediction)[0]

    # Display result
    st.success(f"✅ Recommended Crop: **{recommended_crop}**")
