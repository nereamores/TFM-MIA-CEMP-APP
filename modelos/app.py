import streamlit as st
import joblib
import pandas as pd

# Colores personalizados
COLOR_TITULOS = "#676A73"
COLOR_CEMP = "#E97F87"
paleta_colores = [COLOR_TITULOS, COLOR_CEMP]

# Cargar pipeline
pipeline_path = "modelos/diabetes_rf_pipeline.pkl"
rf_pipeline = joblib.load(pipeline_path)

# Título
st.title("🩺 Predicción de Diabetes")
st.markdown("Introduce los datos del paciente:")

# Inputs dinámicos
feature_names = rf_pipeline.named_steps["imputer"].feature_names_in_
user_input = {}
for col in feature_names:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# Botón de predicción
if st.button("Predecir"):
    X_new = pd.DataFrame([user_input])

    # Variables derivadas
    X_new["Indice_resistencia"] = X_new["Glucose"] * X_new["Insulin"]
    X_new["BMI_square"] = X_new["BMI"] ** 2
    X_new["Is_prediabetes"] = (X_new["Glucose"] >= 140).astype(int)

    # Predicción
    pred_proba = rf_pipeline.predict_proba(X_new)[0,1]
    threshold = 0.27
    pred_class = int(pred_proba >= threshold)

    st.subheader("Predicción")
    st.write(f"➡️ Probabilidad de diabetes: **{pred_proba:.2%}**")
    st.write(f"➡️ Predicción final (threshold {threshold}): **{'Diabetes' if pred_class==1 else 'No Diabetes'}**")
