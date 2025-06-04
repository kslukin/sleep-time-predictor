import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load pre-trained model
model = joblib.load("model.pkl")

# Set page config
st.set_page_config(page_title="Sleep Predictor", layout="centered")

# Title
st.title("💤 Sleep Duration Predictor")
st.markdown("Estimate your sleep duration based on your daily habits.")

# Input sliders
st.subheader("📝 Your Daily Inputs")

workout = st.slider("🟢 Workout Time (hours)", 0.0, 3.0, 1.0)
reading = st.slider("🟢 Reading Time (hours)", 0.0, 3.0, 1.0)
phone = st.slider("🔴 Phone Usage Time (hours)", 0.0, 10.0, 2.0)
work = st.slider("🔴 Work Hours", 0.0, 12.0, 8.0)
caffeine = st.slider("🔴 Caffeine Intake (mg)", 0.0, 300.0, 100.0)
relax = st.slider("🟢 Relaxation Time (hours)", 0.0, 3.0, 1.0)

# Input data for model
input_data = pd.DataFrame(
    [[workout, reading, phone, work, caffeine, relax]],
    columns=["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"]
)

# Predict and display
if st.button("🔍 Predict Sleep Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"🛌 Your predicted sleep duration is **{prediction:.2f} hours**")

    # Radar chart
    st.markdown("### 📊 Your Habits Overview")

    categories = ["Workout", "Reading", "Phone", "Work", "Caffeine", "Relax"]
    values = [workout, reading, phone, work, caffeine, relax]

    # Close radar loop
    categories += [categories[0]]
    values += [values[0]]

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
        title="Radar Chart of Your Inputs",
        height=500
    )

    st.plotly_chart(fig)

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
