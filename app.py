import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Dummy training data — same structure as original
df = pd.read_csv("sleeptime_prediction_dataset.csv")
df = df[(df["SleepTime"] >= 1.0) & (df["SleepTime"] <= 12.0)]

X = df.drop(columns=["SleepTime"])
y = df["SleepTime"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Interface
st.title("😴 Sleep Duration Predictor")
st.markdown("Enter your habits:")

workout = st.slider("Workout Time", 0.0, 3.0, 1.0)
reading = st.slider("Reading Time", 0.0, 3.0, 1.0)
phone = st.slider("Phone Time", 0.0, 10.0, 2.0)
work = st.slider("Work Hours", 0.0, 12.0, 8.0)
caffeine = st.slider("Caffeine Intake", 0.0, 300.0, 100.0)
relax = st.slider("Relaxation Time", 0.0, 3.0, 1.0)

input_data = pd.DataFrame([[workout, reading, phone, work, caffeine, relax]],
    columns=X.columns)

if st.button("Predict Sleep Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"🛌 Predicted sleep duration: **{prediction:.2f} hours**")
