import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model
model = joblib.load("model.pkl")

# Page settings
st.set_page_config(page_title="Sleep Duration Predictor", layout="centered")
st.title("😴 Sleep Duration Predictor")
st.markdown("Estimate your predicted sleep duration based on your daily habits.")

# Sliders with emojis
st.subheader("🧠 Your Daily Inputs")
workout = st.slider("🟢 Workout Time (hours)", 0.0, 3.0, 1.0)
reading = st.slider("🟢 Reading Time (hours)", 0.0, 3.0, 1.0)
phone = st.slider("🔴 Phone Usage Time (hours)", 0.0, 10.0, 2.0)
work = st.slider("🔴 Work Hours", 0.0, 12.0, 8.0)
caffeine = st.slider("🔴 Caffeine Intake (mg)", 0.0, 300.0, 100.0)
relax = st.slider("🟢 Relaxation Time (hours)", 0.0, 3.0, 1.0)

# Prepare input
features = ["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"]
values = [workout, reading, phone, work, caffeine, relax]
input_data = pd.DataFrame([values], columns=features)

# Predict
if st.button("🔍 Predict Sleep Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"🛌 Your predicted sleep duration is **{prediction:.2f} hours**")

    # ----------------------------
    # 1️⃣ STACKED BAR CHART
    # ----------------------------
    import numpy as np

    importances = model.feature_importances_
    contributions = np.array(values) * importances

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=["Sleep Time"],
        y=contributions,
        name="Contributions",
        marker_color=["#A6E3A1", "#B7E4C7", "#FFADAD", "#FFADAD", "#FFADAD", "#A0C4FF"],
        text=features,
        hovertemplate='%{text}: %{y:.2f} hours',
    ))

    fig1.update_layout(
        title="📊 Feature Contributions to Predicted Sleep Time",
        barmode='stack',
        yaxis_title="Hours",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig1)

    # ----------------------------
    # 2️⃣ SLEEP GAUGE
    # ----------------------------
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

    # Insightful feedback
    st.markdown("### 🧾 Interpreting Your Inputs:")
    insights = []

    if caffeine > 200:
        insights.append("☕ Your caffeine intake is quite high and may reduce sleep quality.")
    elif caffeine < 50:
        insights.append("✅ Low caffeine intake is favorable for better sleep.")

    if phone > 5:
        insights.append("📱 High phone usage might delay your sleep onset.")

    if reading >= 1.5:
        insights.append("📘 Great! Reading time is associated with better sleep hygiene.")

    if relax >= 2:
        insights.append("🧘 You're giving yourself enough time to unwind — that helps sleep.")

    if workout < 0.5:
        insights.append("🏃 Consider adding light workouts — it can improve sleep.")

    if not insights:
        st.markdown("Looks like your habits are pretty balanced! ✅")
    else:
        for item in insights:
            st.markdown(f"- {item}")
