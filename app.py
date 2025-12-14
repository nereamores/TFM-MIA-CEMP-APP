import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="CEMP AI Predictor",
    page_icon="🧬",
    layout="wide"
)

# --- COLORES Y ESTILOS ---
CEMP_PINK = "#E97F87"
SUCCESS_GREEN = "#28a745"

# CSS para forzar estilos que Streamlit no deja configurar fácil
st.markdown(f"""
    <style>
    /* Ocultar menú de hamburguesa y footer para look de App nativa */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Estilo del Logo CEMP */
    .cemp-logo {{
        font-family: 'Helvetica', sans-serif;
        font-weight: 900;
        font-size: 3rem;
        color: white;
        margin-bottom: 0px;
        letter-spacing: -2px;
    }}
    .cemp-logo span {{
        color: {CEMP_PINK};
    }}
    
    /* Cajas de métricas autocalculadas */
    .metric-card {{
        background-color: #383838;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
        border: 1px solid #444;
    }}
    .metric-value {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {CEMP_PINK};
    }}
    .metric-label {{
        font-size: 0.8rem;
        color: #bbb;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Estilo personalizado para resultados */
    .result-card-high {{
        background: linear-gradient(135deg, {CEMP_PINK} 0%, #c05e65 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(233, 127, 135, 0.4);
    }}
    </style>
""", unsafe_allow_html=True)

# --- MODELO SIMULADO ---
class MockModel:
    def predict_proba(self, X):
        # Lógica dummy basada en glucosa y BMI
        score = (X[0] * 0.6) + (X[1] * 0.4) # Glucosa + BMI
        prob = 1 / (1 + np.exp(-(score - 100) / 15))
        return [[1-prob, prob]]

if 'model' not in st.session_state: st.session_state.model = MockModel()

# --- BARRA LATERAL (INPUTS CON SLIDERS) ---
with st.sidebar:
    # Logo Estilo CEMP
    st.markdown('<div class="cemp-logo">CEMP<span>.</span>AI</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #888; font-size: 0.8rem; margin-bottom: 30px;'>CLINICAL DECISION SUPPORT SYSTEM</p>", unsafe_allow_html=True)
    
    st.markdown("### 📝 Parámetros Clínicos")
    
    # AQUÍ ESTÁ EL CAMBIO: Usamos st.slider para las "barritas"
    # El color será ROSA automáticamente por el config.toml
    
    glucose = st.slider("Glucosa (mg/dL)", 50, 300, 120, help="Nivel de glucosa en plasma a las 2 horas.")
    bmi = st.slider("BMI (kg/m²)", 15.0, 60.0, 28.5, format="%.1f")
    insulin = st.slider("Insulina (mu U/ml)", 0, 800, 80)
    age = st.slider("Edad (años)", 21, 90, 45)
    
    # Expander para datos menos comunes (para limpiar la vista)
    with st.expander("Ver más parámetros"):
        pregnancies = st.slider("Embarazos", 0, 15, 1)
        blood_pressure = st.slider("Presión Arterial", 40, 140, 70)
        skin_thickness = st.slider("Grosor Piel (mm)", 0, 100, 20)
        dpf = st.slider("Función Pedigrí (DPF)", 0.0, 2.5, 0.47)

    # --- VARIABLES AUTOCALCULADAS (Visualización bonita) ---
    st.markdown("---")
    st.markdown("### 🧬 Métricas Derivadas")
    
    # Cálculos
    res_index = glucose * insulin
    bmi_sq = bmi ** 2
    is_pre = glucose > 140
    
    # Mostrar como tarjetitas usando columnas
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{res_index:.0f}</div>
                <div class="metric-label">Resistencia</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{bmi_sq:.0f}</div>
                <div class="metric-label">BMI²</div>
            </div>
        """, unsafe_allow_html=True)
        
    if is_pre:
        st.markdown(f'<div style="background-color: {CEMP_PINK}; color: white; padding: 5px; border-radius: 5px; text-align: center; font-size: 0.8rem; font-weight: bold; margin-top:5px;">⚠️ PREDIABETES</div>', unsafe_allow_html=True)

# --- ÁREA PRINCIPAL ---

# Lógica
input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
prob = st.session_state.model.predict_proba(input_data)[0][1]
threshold = 0.27
high_risk = prob > threshold

# Título y Pestañas
st.title("Sistema de Predicción")
tab1, tab2, tab3 = st.tabs(["Diagnóstico", "Explicabilidad (SHAP)", "Simulación"])

with tab1:
    col_main_1, col_main_2 = st.columns([1.5, 1])
    
    with col_main_1:
        st.write("") # Espaciador
        if high_risk:
            st.markdown(f"""
                <div class="result-card-high">
                    <h1 style="margin:0; font-size: 2.5rem;">RIESGO ELEVADO</h1>
                    <p style="opacity: 0.9;">Se sugiere intervención inmediata.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
             st.markdown(f"""
                <div style="background-color: #383838; padding: 25px; border-radius: 15px; text-align: center; border: 1px solid #555;">
                    <h1 style="margin:0; font-size: 2.5rem; color: {SUCCESS_GREEN};">Riesgo Bajo</h1>
                    <p style="color: #bbb;">Mantener control rutinario.</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.info("ℹ️ Nota Clínica: Valores de Glucosa > 140 requieren confirmación según guías ADA.")

    with col_main_2:
        # Gauge Chart Minimalista
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        
        # Donut chart
        sizes = [prob, 1-prob]
        colors = [CEMP_PINK if high_risk else SUCCESS_GREEN, '#333333']
        
        ax.pie(sizes, colors=colors, startangle=90, counterclock=False, 
               wedgeprops=dict(width=0.1, edgecolor='none'))
        
        # Texto central
        ax.text(0, 0, f"{prob*100:.1f}%", ha='center', va='center', fontsize=30, fontweight='bold', color='white')
        ax.text(0, -0.3, "Probabilidad", ha='center', va='center', fontsize=10, color='#999')
        
        st.pyplot(fig, use_container_width=True)

with tab2:
    st.markdown("### ¿Qué influye en este paciente?")
    # SHAP Simulado con colores correctos
    features = ["Glucosa", "BMI", "Insulina", "Edad"]
    vals = [0.3, 0.15, 0.05, -0.1]
    
    fig_s, ax_s = plt.subplots(figsize=(8, 3))
    fig_s.patch.set_facecolor('none')
    ax_s.set_facecolor('none')
    
    y = np.arange(len(features))
    colors_shap = [CEMP_PINK if x > 0 else "#555" for x in vals]
    
    ax_s.barh(y, vals, color=colors_shap, edgecolor='none', height=0.6)
    ax_s.set_yticks(y)
    ax_s.set_yticklabels(features, color="white")
    ax_s.spines['bottom'].set_color('#555')
    ax_s.spines['top'].set_visible(False) 
    ax_s.spines['right'].set_visible(False) 
    ax_s.spines['left'].set_visible(False) 
    ax_s.tick_params(axis='x', colors='#888')
    
    st.pyplot(fig_s)

with tab3:
    st.write("Simulación interactiva...")
