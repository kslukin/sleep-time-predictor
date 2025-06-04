import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load pre-trained model
model = joblib.load("model.pkl")

# App title and description
st.set_page_config(page_title="Sleep Predictor", layout="centered")
st.title("😴 Sleep Duration Predictor")
st.markdown("Estimate your sleep duration based on daily habits using a machine learning model.")

# Sidebar for inputs
st.sidebar.header("Your Daily Habits")
workout = st.sidebar.slider("Workout Time (hours)", 0.0, 3.0, 1.0)
reading = st.sidebar.slider("Reading Time (hours)", 0.0, 3.0, 1.0)
phone = st.sidebar.slider("Phone Usage Time (hours)", 0.0, 10.0, 2.0)
work = st.sidebar.slider("Work Hours", 0.0, 12.0, 8.0)
caffeine = st.sidebar.slider("Caffeine Intake (mg)", 0.0, 300.0, 100.0)
relax = st.sidebar.slider("Relaxation Time (hours)", 0.0, 3.0, 1.0)

# Prepare input for model
input_data = pd.DataFrame(
    [[workout, reading, phone, work, caffeine, relax]],
    columns=["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"]
)

# Predict and visualize
if st.button("🧮 Predict Sleep Time"):
    prediction = model.predict(input_data)[0]

    st.subheader("🛌 Estimated Sleep Duration")
    st.success(f"Based on your habits, you may sleep around **{prediction:.2f} hours** tonight.")

    # Plotly bar chart with recommended range
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["You"],
        y=[prediction],
        name="Your Predicted Sleep",
        marker_color="indianred"
    ))

    # Recommended sleep range (7–9h)
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=0.5, y0=7, y1=9,
        fillcolor="LightGreen", opacity=0.3,
        line_width=0,
        layer="below"
    )

    fig.update_layout(
        title="Sleep Time vs. Recommended Range (7–9 hours)",
        yaxis_title="Sleep Duration (hours)",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)
