# ==============================
# Diabetes Prediction - Model Comparison with Hyperparameter Tuning
# ==============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("diabetes.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Hyperparameter Tuning for Random Forest
# ------------------------------
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Hyperparameter Tuning for Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_

# ------------------------------
# Train Other Models Without Tuning
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": best_rf,
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": best_gb,
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    results[name] = {"accuracy": acc, "auc": auc, "model": model}

# ------------------------------
# Streamlit Dashboard
# ------------------------------
st.title("Diabetes Prediction - Model Comparison with Hyperparameter Tuning")

# Show accuracy table
st.subheader("Model Performance Comparison")
perf_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [f"{results[m]['accuracy']*100:.2f}%" for m in results],
    "ROC-AUC": [f"{results[m]['auc']:.2f}" if results[m]['auc'] is not None else "N/A" for m in results]
})
st.dataframe(perf_df)

# Best Model
best_model_name = max(results, key=lambda m: results[m]['accuracy'])
st.success(f"Best Performing Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']*100:.2f}%)")

# Confusion Matrix for Best Model
st.subheader("Confusion Matrix - Best Model")
best_model = results[best_model_name]["model"]
y_best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_best_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ------------------------------
# Save Best Model
# ------------------------------
joblib.dump(best_model, "best_diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ------------------------------
# Prediction Function
# ------------------------------
def predict_diabetes(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    scaler_loaded = joblib.load("scaler.pkl")
    model_loaded = joblib.load("best_diabetes_model.pkl")
    input_scaled = scaler_loaded.transform(input_array)
    prediction = model_loaded.predict(input_scaled)
    return "Diabetes: Yes" if prediction[0] == 1 else "Diabetes: No"

# ------------------------------
# Streamlit Input Form
# ------------------------------
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
