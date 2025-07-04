import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- Page config ---
st.set_page_config(page_title="Sleep Duration Predictor", layout="centered")

# --- Background image and styled container ---
st.markdown(
    """
    <style>
    /* Background image */
    .stApp {
        background-image: url("https://raw.githubusercontent.com/kslukin/sleep-time-predictor/main/Background_2.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* White semi-transparent card for inputs */
    .main > div {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
    }
    /* Remove white footer or ghost block */
    .st-emotion-cache-zq5wmm, footer {
        display: none !important;
        height: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
    }

    html, body, .main {
        overflow-x: hidden;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }  

    h1, h2, h3 {
        color: #1f4e79;
        text-align: center;
    }
    
    /* Remove white footer/padding */
    footer, .css-zq5wmm {
        visibility: hidden;
        height: 0px;
        padding: 0px;
        margin: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load model ---
model = joblib.load("model.pkl")

# --- Title ---
st.title("💤 Sleep Duration Predictor")
st.markdown("Estimate your predicted sleep duration based on your daily habits.")

# --- Input sliders ---
st.subheader("📝 Your Daily Inputs")
workout = st.slider("🟢 Workout Time (hours)", 0.0, 3.0, 1.0)
reading = st.slider("🟢 Reading Time (hours)", 0.0, 3.0, 1.0)
phone = st.slider("🔴 Phone Usage Time (hours)", 0.0, 10.0, 2.0)
work = st.slider("🔴 Work Hours", 0.0, 12.0, 8.0)
caffeine = st.slider("🔴 Caffeine Intake (mg)", 0.0, 300.0, 100.0)
relax = st.slider("🟢 Relaxation Time (hours)", 0.0, 3.0, 1.0)

# --- Predict ---
features = ["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"]
values = [workout, reading, phone, work, caffeine, relax]
input_data = pd.DataFrame([values], columns=features)

if st.button("🔍 Predict Sleep Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"🛌 Your predicted sleep duration is **{prediction:.2f} hours**")

    # --- Sleep Gauge ---
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sleep Duration (hours)"},
        gauge={
            'axis': {'range': [0, 12]},
            'bar': {'color': "royalblue"},
            'steps': [
                {'range': [0, 5], 'color': "#FF6B6B"},
                {'range': [5, 7], 'color': "#FFD93D"},
                {'range': [7, 9], 'color': "#A3F7BF"},
                {'range': [9, 12], 'color': "#B3CDE0"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.8,
                'value': prediction
            }
        }
    ))
    fig2.update_layout(height=350, margin=dict(t=30, b=10, l=30, r=30))
    st.plotly_chart(fig2)

    # --- Feedback block ---
    st.markdown("### 🔍 Insight Based on Your Habits")
    feedback = []

    if caffeine > 200:
        feedback.append("☕ High caffeine intake might reduce sleep quality.")
    elif caffeine < 50:
        feedback.append("✅ Low caffeine intake supports better sleep.")

    if phone > 5:
        feedback.append("📱 High phone usage can delay falling asleep.")

    if reading >= 1.5:
        feedback.append("📘 Great! Reading improves sleep hygiene.")

    if relax >= 2:
        feedback.append("🧘 Relaxation before bed is beneficial.")

    if workout < 0.5:
        feedback.append("🏃 Consider adding light exercise to improve sleep.")

    if not feedback:
        st.markdown("✅ Your lifestyle looks well-balanced!")
    else:
        for point in feedback:
            st.markdown(f"- {point}")
