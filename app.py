import streamlit as st
from src.data_preprocessing import load_data, scale_and_split
from src.train_models import train_models
from src.evaluate_models import evaluate_models, best_model
from src.predict import predict_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load and preprocess data
X, y = load_data("diabetes.csv")
X_train, X_test, y_train, y_test, scaler = scale_and_split(X, y)

# Train and evaluate models
models = train_models(X_train, y_train)
results = evaluate_models(models, X_train, y_train, X_test, y_test)
best_model_name = best_model(results)

# Save best model and scaler
joblib.dump(results[best_model_name]["model"], "models/best_diabetes_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Streamlit UI
st.title("Diabetes Prediction")
st.subheader("Enter Patient Details")

col1, col2, col3 = st.columns(3)
with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
with col2:
    skin = st.number_input("Skin Thickness", 0, 100, 30)
    insulin = st.number_input("Insulin", 0, 900, 100)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
with col3:
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 0, 120, 32)

if st.button("Predict"):
    input_data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
    if glucose == 0 or bmi == 0:
        st.warning("Glucose and BMI cannot be zero!")
    else:
        result = predict_diabetes(input_data)
        st.success(result)
