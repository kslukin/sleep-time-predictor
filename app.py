import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Sleep Duration Predictor", layout="wide")

# Load model
model = joblib.load("model.pkl")

# Hide Streamlit footer and menu
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.title("😴 Sleep Duration Predictor")
st.markdown("Estimate your ideal sleep time based on your daily activities.")

# Input layout
st.subheader("🧠 Your Daily Habits")

col1, col2, col3 = st.columns(3)

with col1:
    workout = st.slider("Workout Time (hours)", 0.0, 3.0, 1.0, help="How long you exercised today")
    work = st.slider("Work Hours", 0.0, 12.0, 8.0, help="Total hours spent working")

with col2:
    reading = st.slider("Reading Time (hours)", 0.0, 3.0, 1.0, help="Time spent reading books or articles")
    caffeine = st.slider("Caffeine Intake (mg)", 0.0, 300.0, 100.0, help="Approximate caffeine consumed")

with col3:
    phone = st.slider("Phone Usage Time (hours)", 0.0, 10.0, 2.0, help="Total screen time on phone")
    relax = st.slider("Relaxation Time (hours)", 0.0, 3.0, 1.0, help="Time spent on relaxing activities")

# Prepare input for prediction
input_data = pd.DataFrame(
    [[workout, reading, phone, work, caffeine, relax]],
    columns=["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"]
)

# Predict
if st.button("🧮 Predict Sleep Time"):
    prediction = model.predict(input_data)[0]
    st.subheader("🛌 Estimated Sleep Duration")
    st.success(f"Based on your habits, you may sleep around **{prediction:.2f} hours** tonight.")
