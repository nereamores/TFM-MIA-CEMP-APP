import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="CEMP AI", page_icon="🏥", layout="wide")

# --- COLORES CEMP ---
CEMP_PINK = "#E97F87"
SUCCESS_GREEN = "#28a745"
TEXT_DARK = "#333333"

# --- ESTILOS CSS (MODO CLARO ENTERPRISE) ---
st.markdown(f"""
    <style>
    /* Ocultar elementos extra */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* LOGO CEMP (Estilo oscuro para fondo claro) */
    .cemp-logo {{
        font-family: 'Arial', sans-serif; font-weight: 900; font-size: 3rem;
        color: {TEXT_DARK}; margin-bottom: 0px; line-height: 1;
    }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}
    .cemp-subtitle {{ color: #666; font-size: 0.8rem; margin-bottom: 30px; letter-spacing: 1px; }}
    
    /* TARJETAS DE MÉTRICAS (Sidebar) - Estilo limpio y blanco */
    .metric-card {{
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #E0E0E0; /* Borde sutil */
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); /* Sombra muy suave */
    }}
    .metric-value {{ font-size: 1.6rem; font-weight: bold; color: {CEMP_PINK}; }}
    .metric-label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; font-weight: 600; }}
    
    /* CAJAS DE RESULTADO PRINCIPAL (Estilo Enterprise) */
    .result-box {{
        padding: 30px; border-radius: 15px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        background-color: #FFFFFF; /* Fondo blanco limpio */
    }}
    .result-high {{
        border-top: 5px solid {CEMP_PINK}; /* Línea superior rosa */
    }}
    .result-low {{
        border-top: 5px solid {SUCCESS_GREEN}; /* Línea superior verde */
    }}
    .result-title {{ font-size: 2.2rem; font-weight: bold; margin: 0; }}
    .result-text {{ font-size: 1.1rem; margin-top: 15px; color: #555; }}

    /* Estilo de pestaña activa */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        color: {CEMP_PINK} !important;
        border-bottom-color: {CEMP_PINK} !important;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# --- MODELO SIMULADO ---
if 'model' not in st.session_state:
    class MockModel:
        def predict_proba(self, X):
            score = (X[0] * 0.5) + (X[1] * 0.5) # Glucosa y BMI
            prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

# --- BARRA LATERAL (BLANCA) ---
with st.sidebar:
    st.markdown('<div class="cemp-logo">CEMP<span>.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="cemp-subtitle">PLATAFORMA DE SALUD DE PRECISIÓN</div>', unsafe_allow_html=True)
    
    st.markdown("### 📝 Parámetros Clínicos")
    # Sliders (se verán rositas por el config.toml)
    glucose = st.slider("Glucosa (mg/dL)", 50, 250, 120)
    bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 24.5)
    insulin = st.slider("Insulina (mu U/ml)", 0, 600, 80)
    age = st.slider("Edad (años)", 18, 90, 35)
    
    with st.expander("Más opciones"):
        pregnancies = st.slider("Embarazos", 0, 15, 0)
        dpf = st.slider("Función Pedigrí", 0.0, 2.5, 0.5)

    st.markdown("---")
    st.markdown("### 📊 Métricas Calculadas")
    
    res_index = glucose * insulin
    bmi_sq = bmi ** 2
    
    # Tarjetas blancas limpias para las métricas
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{res_index:.0f}</div><div class="metric-label">Resistencia (G×I)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{bmi_sq:.0f}</div><div class="metric-label">Índice BMI²</div></div>', unsafe_allow_html=True)

# --- PANEL PRINCIPAL (FONDO GRIS CLARO) ---
st.title("Tablero de Predicción de Diabetes")
st.markdown("Análisis de riesgo basado en inteligencia artificial.")

input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
prob = st.session_state.model.predict_proba(input_data)[0][1]
is_high_risk = prob > 0.27

tab1, tab2, tab3 = st.tabs(["🏥 Diagnóstico", "🔍 Explicabilidad", "🎛️ Simulación"])

with tab1:
    st.write("") # Espacio
    col_res, col_gauge = st.columns([2, 1], gap="large")
    
    with col_res:
        if is_high_risk:
            # Tarjeta blanca con acento ROSA
            st.markdown(f"""
                <div class="result-box result-high">
                    <h2 class="result-title" style="color: {CEMP_PINK};">ALTO RIESGO DETECTADO</h2>
                    <p class="result-text">La probabilidad calculada es del <strong>{prob*100:.1f}%</strong>. Se recomienda iniciar protocolo de seguimiento.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Tarjeta blanca con acento VERDE
            st.markdown(f"""
                <div class="result-box result-low">
                    <h2 class="result-title" style="color: {SUCCESS_GREEN};">BAJO RIESGO</h2>
                    <p class="result-text">El paciente se mantiene en rangos saludables. Continuar control rutinario.</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.info("ℹ️ **Nota Clínica:** Valores de Glucosa en ayunas > 126 mg/dL requieren confirmación adicional según guías ADA.")

    with col_gauge:
        # Gráfico Donut limpio para fondo claro
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('none') # Transparente
        ax.set_facecolor('none')
        
        color_accent = CEMP_PINK if is_high_risk else SUCCESS_GREEN
        # Anillo exterior gris claro, relleno de color
        ax.pie([prob, 1-prob], colors=[color_accent, '#E0E0E0'], startangle=90, counterclock=False, 
               wedgeprops=dict(width=0.12, edgecolor='white', linewidth=2))
        
        # Texto central oscuro
        ax.text(0, 0.1, f"{prob*100:.0f}%", ha='center', va='center', fontsize=35, fontweight='bold', color=TEXT_DARK)
        ax.text(0, -0.25, "Probabilidad", ha='center', va='center', fontsize=12, color='#999')
        
        st.pyplot(fig, use_container_width=True)

with tab2:
    st.subheader("Análisis de Factores (SHAP)")
    # Simulación de gráfico SHAP para fondo claro
    features = ["Glucosa", "BMI", "Edad", "Insulina"]
    vals = [0.3, 0.15, -0.1, 0.05]
    
    fig_s, ax_s = plt.subplots(figsize=(8, 3))
    fig_s.patch.set_facecolor('none')
    ax_s.set_facecolor('none')
    
    y = np.arange(len(features))
    # Barras Rosas para riesgo, Gris oscuro para salud
    colors_shap = [CEMP_PINK if x > 0 else "#888" for x in vals]
    
    ax_s.barh(y, vals, color=colors_shap, height=0.6)
    ax_s.set_yticks(y)
    ax_s.set_yticklabels(features, color=TEXT_DARK, fontsize=11)
    ax_s.axvline(0, color='#ddd', linewidth=1) # Línea central sutil
    ax_s.spines['bottom'].set_visible(False)
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)
    ax_s.spines['left'].set_visible(False)
    ax_s.tick_params(axis='x', colors='#888')
    
    st.pyplot(fig_s)

with tab3:
    st.write("Simulación interactiva disponible próximamente.")
