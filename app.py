import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Sleep Predictor", layout="centered")

st.title("😴 Sleep Duration Predictor")
st.markdown("Enter your daily habits below to predict how much you'll sleep tonight.")

# Input sliders for user features
workout = st.slider("Workout Time (hours/day)", 0.0, 3.0, 1.0, step=0.1)
reading = st.slider("Reading Time (hours/day)", 0.0, 3.0, 0.5, step=0.1)
phone = st.slider("Phone Usage (hours/day)", 0.0, 10.0, 2.0, step=0.1)
work = st.slider("Work Hours (hours/day)", 0.0, 12.0, 8.0, step=0.1)
caffeine = st.slider("Caffeine Intake (mg/day)", 0.0, 300.0, 100.0, step=5.0)
relax = st.slider("Relaxation Time (hours/day)", 0.0, 3.0, 1.0, step=0.1)

# Make prediction
if st.button("Predict Sleep Time"):
    input_data = pd.DataFrame([[workout, reading, phone, work, caffeine, relax]],
                              columns=["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"])
    
    prediction = model.predict(input_data)[0]

    st.subheader("Predicted Sleep Duration:")
    st.success(f"🕒 You are likely to sleep **{prediction:.2f} hours**.")
