import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model
model = joblib.load("model.pkl")

# Title and description
st.title("Sleep Duration Predictor")
st.markdown("Adjust your daily activities to see how they influence your sleep time.")

# Custom CSS for colored sliders
st.markdown("""
<style>
.green .stSlider > div > div > div { background: #a4d4ae; }
.red .stSlider > div > div > div { background: #f5a3a3; }
</style>
""", unsafe_allow_html=True)

# Colored sliders
with st.container():
    st.markdown("**Lifestyle Inputs:**")

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown('<div class="green">', unsafe_allow_html=True)
            workout = st.slider("Workout Time (hours)", 0.0, 3.0, 1.0, key="w1")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="green">', unsafe_allow_html=True)
            reading = st.slider("Reading Time (hours)", 0.0, 3.0, 1.0, key="r1")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="red">', unsafe_allow_html=True)
            phone = st.slider("Phone Time (hours)", 0.0, 10.0, 2.0, key="p1")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="red">', unsafe_allow_html=True)
            work = st.slider("Work Hours", 0.0, 12.0, 8.0, key="w2")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="red">', unsafe_allow_html=True)
            caffeine = st.slider("Caffeine Intake (mg)", 0.0, 300.0, 100.0, key="c1")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="green">', unsafe_allow_html=True)
            relax = st.slider("Relaxation Time (hours)", 0.0, 3.0, 1.0, key="rl1")
            st.markdown('</div>', unsafe_allow_html=True)

# Make prediction
input_data = pd.DataFrame([[workout, reading, phone, work, caffeine, relax]],
    columns=["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"])

if st.button("Predict Sleep Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted sleep duration: **{prediction:.2f} hours**")

    # Feature contributions (feature importance × value)
    importance = model.feature_importances_
    contributions = input_data.values[0] * importance
    impact_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Impact": contributions
    }).sort_values(by="Impact", ascending=False)

    fig = px.bar(impact_df, x="Impact", y="Feature", orientation="h",
                 title="Estimated Feature Impact on Predicted Sleep Time",
                 labels={"Impact": "Contribution to Prediction"},
                 template="simple_white")

    st.plotly_chart(fig)
