import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="CEMP Diabetes Predictor",
    page_icon="🏥",
    layout="wide"
)

# --- COLORES CEMP (Para uso en CSS y gráficos) ---
CEMP_PINK = "#FF6B6B"
CEMP_DARK_BG = "#1A1A1A"
CEMP_SIDEBAR_BG = "#262626"
CEMP_GRAY_TEXT = "#B0B0B0"
SUCCESS_GREEN = "#28a745" # Para bajo riesgo

# --- CSS PERSONALIZADO PARA REMATAR EL LOOK CEMP ---
st.markdown(f"""
    <style>
    /* Ajuste fino para inputs en modo oscuro */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {{
        background-color: #333333;
        color: white;
        border: 1px solid #555555;
    }}
    
    /* Estilo para las métricas autocalculadas en el sidebar */
    div[data-testid="stMetricValue"] {{
        font-size: 1.2rem !important;
        color: {CEMP_PINK} !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.9rem !important;
        color: {CEMP_GRAY_TEXT} !important;
    }}

    /* Estilo para el gauge de probabilidad (texto grande) */
    .big-font {{
        font-size: 4rem !important;
        font-weight: bold;
    }}
    
    /* Estilos para las cajas de resultado personalizadas */
    .result-box-high {{
        background-color: {CEMP_PINK};
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }}
    .result-box-low {{
        background-color: #333333; /* Gris oscuro neutro */
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid {SUCCESS_GREEN};
        margin-bottom: 20px;
    }}
    .result-title {{
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }}
    .result-subtitle {{
        font-size: 1.1rem;
        margin-top: 10px;
        font-weight: 300;
    }}
    
    /* Estilo para la nota clínica */
    .clinical-note {{
        background-color: #262626;
        border-left: 5px solid {CEMP_PINK};
        padding: 15px;
        border-radius: 5px;
        color: {CEMP_GRAY_TEXT};
        font-size: 0.95rem;
    }}
    </style>
""", unsafe_allow_html=True)

# --- MODELO SIMULADO (Mock) ---
class MockModel:
    def predict_proba(self, X):
        # Lógica simple para demostración visual
        glucose = X[0]
        bmi = X[1]
        age = X[3]
        # Crear un score basado en reglas simples
        score = 0
        if glucose > 140: score += 2
        if bmi > 30: score += 1.5
        if age > 50: score += 1
        
        # Convertir a una "probabilidad" simulada
        base_prob = 0.15
        prob = min(0.95, base_prob + (score * 0.15))
        return [[1-prob, prob]]

# Inicializar modelo
if 'model' not in st.session_state:
    st.session_state.model = MockModel()

model = st.session_state.model

# --- BARRA LATERAL ---
with st.sidebar:
    # Si tuvieras el logo de CEMP en blanco/rosa, quedaría genial aquí
    # st.image("logo_cemp_white.png", width=150) 
    st.title("Datos del Paciente")
    
    # Inputs médicos
    pregnancies = st.number_input("Embarazos", 0, 20, 1)
    glucose = st.number_input("Glucosa (mg/dL)", 50, 300, 120)
    blood_pressure = st.number_input("Presión Arterial (mm Hg)", 40, 200, 70)
    skin_thickness = st.number_input("Grosor Piel (mm)", 0, 100, 20)
    insulin = st.number_input("Insulina (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 28.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, format="%.2f")
    age = st.number_input("Edad", 18, 120, 35)
    
    # --- VARIABLES AUTOCALCULADAS ---
    st.markdown("---")
    st.subheader("Variables Derivadas")
    
    # 1. Índice de Resistencia (Proxy)
    resistance_index = glucose * insulin
    # 2. BMI Cuadrático
    bmi_squared = bmi ** 2
    # 3. Flag Prediabetes
    is_prediabetes = glucose > 140

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Resistencia (G×I)", f"{resistance_index:.0f}")
    with col_m2:
        st.metric("BMI²", f"{bmi_squared:.0f}")
        
    if is_prediabetes:
        st.markdown(f"""
            <div style="background-color: {CEMP_PINK}; color: white; padding: 8px; border-radius: 6px; text-align: center; font-weight: bold; margin-top: 10px;">
                ⚠️ Marcador Prediabetes: POSITIVO
            </div>
        """, unsafe_allow_html=True)
    else:
         st.markdown(f"""
            <div style="background-color: #333333; color: #B0B0B0; padding: 8px; border-radius: 6px; text-align: center; font-size: 0.9rem; margin-top: 10px;">
                Marcador Prediabetes: Negativo
            </div>
        """, unsafe_allow_html=True)

# --- ÁREA PRINCIPAL ---

# Título principal
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>Sistema de Predicción de Diabetes Mellitus</h1>", unsafe_allow_html=True)

# Pestañas
tab_diag, tab_expl, tab_sim = st.tabs(["🏥 Diagnóstico Clínico", "🔍 Explicabilidad (SHAP)", "🎛️ Simulación Terapéutica"])

# Lógica de predicción
input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
probability = model.predict_proba(input_data)[0][1]
threshold = 0.27
is_high_risk = probability > threshold

with tab_diag:
    col_left, col_right = st.columns([3, 2], gap="large")
    
    with col_left:
        st.subheader("Resultado del Análisis")
        
        if is_high_risk:
            # Caja de ALTO RIESGO (Estilo CEMP Pink)
            st.markdown(f"""
                <div class="result-box-high">
                    <p class="result-title">🔴 ALTO RIESGO DETECTADO</p>
                    <p class="result-subtitle">Se recomienda intervención clínica y pruebas confirmatorias.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Caja de BAJO RIESGO (Estilo Neutro Oscuro)
            st.markdown("""
                <div class="result-box-low">
                    <p class="result-title">🟢 Bajo Riesgo Estimado</p>
                    <p class="result-subtitle">Mantener hábitos saludables y control rutinario.</p>
                </div>
            """, unsafe_allow_html=True)
            
        # Nota Clínica Estilizada
        st.markdown(f"""
            <div class="clinical-note">
                ℹ️ <strong>Nota Clínica (ADA Standards):</strong> Pacientes con glucosa en ayunas > 126 mg/dL o HbA1c > 6.5% deben ser confirmados con una segunda prueba, independientemente del modelo.
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.subheader("Probabilidad del Modelo")
        # Color dinámico para el porcentaje
        prob_color = CEMP_PINK if is_high_risk else SUCCESS_GREEN
        
        st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <p class="big-font" style="color: {prob_color};">{probability*100:.1f}%</p>
                <p style="color: {CEMP_GRAY_TEXT};">Certeza estimada</p>
                <p style="font-size: 0.8rem; color: #666;">Umbral de decisión: {threshold}</p>
            </div>
        """, unsafe_allow_html=True)

# Placeholders para las otras pestañas
with tab_expl:
    st.info("El módulo de explicabilidad SHAP se mostraría aquí con fondo oscuro y barras rosa/gris.")
with tab_sim:
    st.info("El módulo de simulación terapéutica se mostraría aquí.")
