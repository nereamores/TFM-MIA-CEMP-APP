import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="CEMP Diabetes Predictor",
    page_icon="🏥",
    layout="wide"
)

# --- COLORES PARA USO INTERNO Y GRÁFICOS ---
CEMP_PINK = "#E97F87"
DARK_BG = "#1A1A1A"
TEXT_WHITE = "#FFFFFF"
SUCCESS_GREEN = "#28a745"

# --- CSS PERSONALIZADO AVANZADO PARA ESTILO SPLIT-SCREEN ---
st.markdown(f"""
    <style>
    /* =========================================
       ESTILOS DE LA BARRA LATERAL (FONDO ROSA)
    ========================================= */
    
    /* Logo CEMP en blanco */
    .sidebar-logo {{
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: white;
        text-align: left;
        margin-bottom: 20px;
    }}

    /* Forzar texto blanco en TODA la sidebar */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div {{
        color: #FFFFFF !important;
    }}

    /* Estilo de los inputs para que parezcan del login (fondos blancos limpios) */
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stSelectbox select {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #333333 !important; /* Texto oscuro dentro del input blanco */
        border: none;
        border-radius: 8px;
        padding: 10px;
    }}
    
    /* Separadores blancos tenues */
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255, 255, 255, 0.4) !important;
    }}

    /* Métricas autocalculadas en blanco brillante */
    [data-testid="stSidebar"] div[data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        color: white !important;
        font-weight: 700;
    }}
    [data-testid="stSidebar"] div[data-testid="stMetricLabel"] {{
        color: rgba(255, 255, 255, 0.8) !important;
    }}
    
    /* Alerts en sidebar */
    .sidebar-alert-danger {{
        background-color: white;
        color: {CEMP_PINK};
        padding: 10px; border-radius: 8px; font-weight: bold; text-align: center;
    }}
     .sidebar-alert-success {{
        background-color: rgba(255,255,255,0.2);
        color: white;
        padding: 10px; border-radius: 8px; font-weight: bold; text-align: center;
        border: 1px solid white;
    }}

    /* =========================================
       ESTILOS DEL ÁREA PRINCIPAL (FONDO OSCURO)
    ========================================= */
    
    /* Título principal */
    .main-title {{
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }}
    
    /* Cajas de resultado */
    .result-box-high {{
        background: linear-gradient(135deg, {CEMP_PINK}, #d06068);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(233, 127, 135, 0.3);
    }}
    
    /* Pestañas activas (subrayado rosa) */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom-color: {CEMP_PINK} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- MODELO MOCK ---
class MockModel:
    def predict_proba(self, X):
        glucose = X[0]
        bmi = X[1]
        age = X[3]
        score = 0
        if glucose > 140: score += 2
        if bmi > 30: score += 1.5
        if age > 50: score += 1
        base_prob = 0.15
        prob = min(0.95, base_prob + (score * 0.15))
        return [[1-prob, prob]]

if 'model' not in st.session_state: st.session_state.model = MockModel()
model = st.session_state.model

# --- BARRA LATERAL (IZQUIERDA - ROSA SÓLIDO) ---
with st.sidebar:
    # Logo simulado CEMP en texto blanco grande
    st.markdown('<div class="sidebar-logo">CEMP</div>', unsafe_allow_html=True)
    st.markdown("### Datos del Paciente")
    
    # Inputs (se verán como cajas blancas sobre fondo rosa)
    pregnancies = st.number_input("Embarazos", 0, 20, 1)
    glucose = st.number_input("Glucosa (mg/dL)", 50, 300, 140)
    blood_pressure = st.number_input("Presión Arterial", 40, 200, 70)
    skin_thickness = st.number_input("Grosor Piel", 0, 100, 20)
    insulin = st.number_input("Insulina (mu U/ml)", 0, 900, 85)
    bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 28.5, format="%.1f")
    dpf = st.number_input("DPF", 0.0, 3.0, 0.5, format="%.2f")
    age = st.number_input("Edad", 18, 120, 45)
    
    # Variables Autocalculadas
    st.markdown("---")
    st.markdown("### Variables Derivadas")
    
    resistance_index = glucose * insulin
    bmi_squared = bmi ** 2
    is_prediabetes = glucose > 140

    col_m1, col_m2 = st.columns(2)
    with col_m1: st.metric("Resistencia (G×I)", f"{resistance_index:.0f}")
    with col_m2: st.metric("BMI²", f"{bmi_squared:.0f}")
        
    st.write("") # Espacio
    if is_prediabetes:
        st.markdown(f'<div class="sidebar-alert-danger">⚠️ Prediabetes: POSITIVO</div>', unsafe_allow_html=True)
    else:
         st.markdown(f'<div class="sidebar-alert-success">Prediabetes: Negativo</div>', unsafe_allow_html=True)

# --- ÁREA PRINCIPAL (DERECHA - OSCURA) ---

st.markdown('<h1 class="main-title">Sistema de Predicción de Diabetes Mellitus</h1>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; opacity: 0.8;'>Suite de Inteligencia Artificial para soporte a la decisión clínica.</p>", unsafe_allow_html=True)
st.write("")

# Lógica de predicción
input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
probability = model.predict_proba(input_data)[0][1]
threshold = 0.27
is_high_risk = probability > threshold

# Pestañas
tab_diag, tab_expl, tab_sim = st.tabs(["🏥 Diagnóstico", "🔍 Explicabilidad", "🎛️ Simulación"])

with tab_diag:
    col_left, col_right = st.columns([3, 2], gap="large")
    
    with col_left:
        if is_high_risk:
            st.markdown(f"""
                <div class="result-box-high">
                    <h2 style="color: white; margin:0; font-size: 2rem;">🔴 ALTO RIESGO DETECTADO</h2>
                    <p style="font-size: 1.2rem; margin-top: 10px;">Se recomienda intervención clínica inmediata.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
             st.markdown(f"""
                <div style="background-color: #333333; padding: 25px; border-radius: 15px; text-align: center; border: 2px solid {SUCCESS_GREEN};">
                    <h2 style="color: {SUCCESS_GREEN}; margin:0;">🟢 Bajo Riesgo Estimado</h2>
                    <p style="margin-top: 10px;">Mantener hábitos saludables y control.</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
            <div style="margin-top: 20px; background-color: #262626; padding: 15px; border-left: 4px solid {CEMP_PINK}; border-radius: 4px; font-size: 0.9rem;">
                ℹ️ <strong>Nota Guías ADA:</strong> Valores de glucosa > 140 mg/dL en OGTT requieren seguimiento independientemente del modelo.
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Gauge de probabilidad minimalista
        prob_color = CEMP_PINK if is_high_risk else SUCCESS_GREEN
        
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rorigin(-1.5)
        ax.set_axis_off()
        
        # Fondo gris
        ax.barh(0, np.radians(180), color='#333333', height=1)
        # Valor
        ax.barh(0, np.radians(probability * 180), color=prob_color, height=1)
        
        # Texto central
        plt.text(0, -1.8, f"{probability*100:.1f}%", ha='center', va='center', fontsize=35, fontweight='bold', color='white')
        plt.text(0, -1.2, "Probabilidad", ha='center', va='center', fontsize=12, color='gray')
        
        fig.patch.set_facecolor('none') # Fondo transparente para matplotlib
        st.pyplot(fig, use_container_width=True)

# --- PESTAÑA EXPLICABILIDAD (SHAP MOCK con colores CEMP) ---
with tab_expl:
    st.subheader("Análisis de Contribución (SHAP Waterfall)")
    # Datos Mock simulando SHAP
    features = ["Glucosa", "BMI", "Edad", "Insulina", "DPF"]
    values = [0.25, 0.15, -0.10, 0.05, -0.02] # + es riesgo (rosa), - es salud (gris)
    base = 0.20
    
    fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
    fig_shap.patch.set_facecolor('none')
    ax_shap.set_facecolor('none')

    y_pos = np.arange(len(features))
    current = base
    for i, (feat, val) in enumerate(zip(features, values)):
        color = CEMP_PINK if val > 0 else "#555555" # Rosa vs Gris oscuro
        ax_shap.barh(i, val, left=current, color=color, edgecolor='none', height=0.5)
        # Etiqueta
        text_pos = current + val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax_shap.text(text_pos, i, f"{val:+.2f}", va='center', ha=ha, color='white')
        current += val

    ax_shap.set_yticks(y_pos)
    ax_shap.set_yticklabels(features, color='white')
    ax_shap.tick_params(axis='x', colors='white')
    ax_shap.spines['bottom'].set_color('white')
    ax_shap.spines['top'].set_visible(False)
    ax_shap.spines['right'].set_visible(False)
    ax_shap.spines['left'].set_visible(False)
    st.pyplot(fig_shap)

with tab_sim:
    st.info("Módulo de Simulación Terapéutica disponible en versión completa.")
