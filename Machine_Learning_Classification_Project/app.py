import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("model.pkl")

# Title
st.title("ML Model Deployment - Prediction App")

st.write("Enter the feature values below to get a prediction:")

# Create input fields
RMS_Voltage = st.number_input("RMS Voltage", value=0.0)
Peak_Voltage = st.number_input("Peak Voltage", value=0.0)
THD = st.number_input("THD", value=0.0)
Duration_ms = st.number_input("Duration (ms)", value=0.0)
DWT_Energy_Level1 = st.number_input("DWT Energy Level 1", value=0.0)
DWT_Energy_Level2 = st.number_input("DWT Energy Level 2", value=0.0)
DWT_Entropy = st.number_input("DWT Entropy", value=0.0)
Signal_Noise_Ratio_dB = st.number_input("Signal to Noise Ratio (dB)", value=0.0)
Phase_A = st.number_input("Phase A", value=0.0)
Phase_B = st.number_input("Phase B", value=0.0)
Phase_C = st.number_input("Phase C", value=0.0)

# When button is clicked
if st.button("Predict"):
    # Put input values into a dataframe
    input_data = pd.DataFrame({
        'RMS_Voltage': [RMS_Voltage],
        'Peak_Voltage': [Peak_Voltage],
        'THD': [THD],
        'Duration_ms': [Duration_ms],
        'DWT_Energy_Level1': [DWT_Energy_Level1],
        'DWT_Energy_Level2': [DWT_Energy_Level2],
        'DWT_Entropy': [DWT_Entropy],
        'Signal_Noise_Ratio_dB': [Signal_Noise_Ratio_dB],
        'Phase_A': [Phase_A],
        'Phase_B': [Phase_B],
        'Phase_C': [Phase_C]
    })

    # Make prediction
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")
