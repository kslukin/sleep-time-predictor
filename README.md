# sleep-time-predictor

# 💤 Sleep Time Predictor

A machine learning app that predicts sleep duration based on daily habits using a synthetic dataset.

## 📌 Overview

This project demonstrates a complete ML pipeline — from data analysis to deployment — using a synthetic dataset simulating the relationship between lifestyle factors and sleep duration.

Live app: [Sleep Time Predictor on Streamlit](https://sleep-time-predictor-shcrdtk9wjtmlqryxecfgn.streamlit.app)

Dataset source: [Kaggle – Sleep Time Prediction](https://www.kaggle.com/datasets/govindaramsriram/sleep-time-prediction)

## 🧠 Project Steps

1. **Data Analysis**  
   Explored and visualized correlations between features such as workout time, phone usage, caffeine intake, and sleep duration.

2. **Preprocessing**  
   Removed outliers, filtered unrealistic values, and selected relevant features.

3. **Model Training**  
   Trained a `RandomForestRegressor` (MAE: 0.24, R²: 0.82) on the cleaned dataset.  
   Model was serialized using `joblib`.

4. **App Development**  
   Built an interactive web interface with [Streamlit](https://streamlit.io) allowing users to input their daily habits and receive a real-time sleep prediction.

5. **Deployment**  
   Hosted the app on Streamlit Cloud and embedded into a personal portfolio at [kslukin.github.io](https://kslukin.github.io).

## 🗃 Files

- `train_model.py` — model training script  
- `model.pkl` — trained and saved model  
- `app.py` — Streamlit app for prediction  
- `requirements.txt` — project dependencies  
- `Data_analysis.ipynb` — EDA and visualization notebook

## 🧪 Tech Stack

- Python, Pandas, Scikit-learn, Streamlit, Joblib, Matplotlib, Seaborn, Plotly  
- Deployment: Streamlit Cloud  
- Data: synthetic Kaggle dataset

## 📈 Future Ideas

- Add model confidence intervals  
- Extend to real-world data integration  
- Provide lifestyle recommendations based on prediction

---

