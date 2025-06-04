import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load pre-trained model
model = joblib.load("model.pkl")

# App title
st.title("💤 Sleep Duration Predictor")
st.markdown("Enter your daily habits below to estimate your predicted sleep duration.")

st.markdown("---")
st.subheader("📝 Your Daily Routine Inputs")

# Sliders with emoji indicators (🟢 positive for sleep, 🔴 negative)
workout = st.slider("🟢 Workout Time (hours)", 0.0, 3.0, 1.0)
reading = st.slider("🟢 Reading Time (hours)", 0.0, 3.0, 1.0)
phone = st.slider("🔴 Phone Usage Time (hours)", 0.0, 10.0, 2.0)
work = st.slider("🔴 Work Hours", 0.0, 12.0, 8.0)
caffeine = st.slider("🔴 Caffeine Intake (mg)", 0.0, 300.0, 100.0)
relax = st.slider("🟢 Relaxation Time (hours)", 0.0, 3.0, 1.0)

# Prepare input
input_data = pd.DataFrame(
    [[workout, reading, phone, work, caffeine, relax]],
    columns=["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"]
)

# Prediction
if st.button("🔍 Predict Sleep Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"🛌 Your predicted sleep duration is **{prediction:.2f} hours**")

    # Radar plot of current input
    st.markdown("### 📊 Your Habits Overview")
    categories = ["Workout", "Reading", "Phone", "Work", "Caffeine", "Relax"]
    values = [workout, reading, phone, work, caffeine, relax]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your habits',
        line=dict(color='royalblue')
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=500,
        title="Radar Chart of Your Inputs"
    )

    st.plotly_chart(fig)
