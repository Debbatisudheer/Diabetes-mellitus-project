import numpy as np
import joblib

def predict_diabetes(input_data, model_path="models/best_diabetes_model.pkl", scaler_path="models/scaler.pkl"):
    input_array = np.array(input_data).reshape(1, -1)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    return "Diabetes: Yes" if prediction[0] == 1 else "Diabetes: No"
