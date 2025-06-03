import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model
model = joblib.load("model.pkl")

# App title
st.title("😴 Sleep Duration Predictor")
st.markdown("Enter your daily habits below to estimate your sleep time:")

# Input sliders
workout = st.slider("Workout Time (hours)", 0.0, 3.0, 1.0)
reading = st.slider("Reading Time (hours)", 0.0, 3.0, 1.0)
phone = st.slider("Phone Usage Time (hours)", 0.0, 10.0, 2.0)
work = st.slider("Work Hours", 0.0, 12.0, 8.0)
caffeine = st.slider("Caffeine Intake (mg)", 0.0, 300.0, 100.0)
relax = st.slider("Relaxation Time (hours)", 0.0, 3.0, 1.0)

# Prepare input for model
input_data = pd.DataFrame(
    [[workout, reading, phone, work, caffeine, relax]],
    columns=["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"]
)

# Predict button
if st.button("Predict Sleep Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"🛌 Predicted sleep duration: **{prediction:.2f} hours**")


