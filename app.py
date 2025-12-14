import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="CEMP Precision Care", page_icon="🩺", layout="wide")

# --- COLORES CEMP & ESTILOS ---
CEMP_PINK = "#E97F87"
CEMP_LIGHT_PINK = "#FADBDD"
SUCCESS_GREEN = "#28a745"
WARNING_YELLOW = "#FFC107"
TEXT_DARK = "#333333"

st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* TIPOGRAFÍA Y LOGO */
    .cemp-logo {{ font-family: 'Arial', sans-serif; font-weight: 900; font-size: 2.8rem; color: {TEXT_DARK}; line-height: 1; }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}
    
    /* TARJETAS DE MÉTRICAS (SIDEBAR) */
    .metric-card {{
        background-color: white; border: 1px solid #eee; padding: 15px; 
        border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px;
    }}
    .metric-val {{ font-size: 1.4rem; font-weight: bold; color: {CEMP_PINK}; }}
    .metric-lbl {{ font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }}

    /* CONTENEDORES PRINCIPALES */
    .main-card {{
        background-color: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #f0f0f0; margin-bottom: 20px;
    }}
    
    /* BARRAS DE PROGRESO PERSONALIZADAS (HTML/CSS) */
    .progress-container {{ width: 100%; background-color: #f1f1f1; border-radius: 5px; margin-top: 5px; }}
    .progress-bar {{ height: 10px; border-radius: 5px; text-align: center; line-height: 10px; color: white; }}
    
    /* RECOMENDACIONES */
    .rec-box {{
        border-left: 4px solid {CEMP_PINK}; background-color: #FFF9F9; padding: 15px; border-radius: 0 10px 10px 0; margin-bottom: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# --- MODELO MOCK ---
if 'model' not in st.session_state:
    class MockModel:
        def predict_proba(self, X):
            score = (X[0] * 0.4) + (X[1] * 0.4) + (X[3] * 0.2) 
            prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

# --- FUNCIONES AUXILIARES DE VISUALIZACIÓN ---
def draw_custom_gauge(prob, title):
    fig, ax = plt.subplots(figsize=(3, 2)) # Más pequeño y compacto
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    color = CEMP_PINK if prob > 0.27 else SUCCESS_GREEN
    
    # Semicírculo
    ax.pie([prob, 1-prob], colors=[color, '#eee'], startangle=90, counterclock=False, 
           wedgeprops=dict(width=0.3, edgecolor='white'))
    
    # Texto
    ax.text(0, -0.3, f"{prob*100:.1f}%", ha='center', va='center', fontsize=24, fontweight='bold', color=TEXT_DARK)
    ax.text(0, -0.7, title, ha='center', va='center', fontsize=9, color='#888')
    
    return fig

# --- SIDEBAR (INPUTS) ---
with st.sidebar:
    st.markdown('<div class="cemp-logo">CEMP<span>.</span>AI</div>', unsafe_allow_html=True)
    st.markdown("PLATAFORMA DE MEDICINA DE PRECISIÓN", unsafe_allow_html=True)
    st.write("")
    
    # Inputs con Sliders Rosas (Gracias al config.toml)
    glucose = st.slider("Glucosa (mg/dL)", 50, 250, 120)
    bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 28.5)
    insulin = st.slider("Insulina (mu U/ml)", 0, 600, 100)
    age = st.slider("Edad", 18, 90, 45)
    
    with st.expander("Factores Secundarios"):
        pregnancies = st.slider("Embarazos", 0, 15, 1)
        dpf = st.slider("Función Pedigrí", 0.0, 2.5, 0.5)
        bp = st.slider("Presión Arterial", 50, 180, 75)

    st.markdown("---")
    
    # Métricas "Glanceable" (Vistazo rápido)
    res_idx = glucose * insulin / 405 # HOMA-IR fórmula real aproximada
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{res_idx:.1f}</div><div class="metric-lbl">HOMA-IR</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{bmi:.1f}</div><div class="metric-lbl">BMI ACTUAL</div></div>', unsafe_allow_html=True)

# --- ÁREA PRINCIPAL ---
input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
prob = st.session_state.model.predict_proba(input_data)[0][1]
is_high = prob > 0.27

# Cabecera con datos del paciente (Simulación de ficha clínica)
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("Informe de Riesgo Metabólico")
    st.markdown(f"**Paciente ID:** #98221 • **Fecha:** 14 Dic 2025 • **Protocolo:** Diabetes T2 Predictor")
with col_head2:
    if is_high:
        st.markdown(f'<div style="background:{CEMP_PINK}; color:white; padding:10px; border-radius:8px; text-align:center; font-weight:bold;">ALTO RIESGO DETECTADO</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:{SUCCESS_GREEN}; color:white; padding:10px; border-radius:8px; text-align:center; font-weight:bold;">BAJO RIESGO</div>', unsafe_allow_html=True)

# Pestañas para organizar la densidad de información
tab1, tab2, tab3 = st.tabs(["📊 Análisis Integral", "🧬 Factores Clínicos", "🤖 Plan de Acción (AI)"])

# --- TAB 1: DASHBOARD VISUAL ---
with tab1:
    c_main, c_pop = st.columns([1.5, 2], gap="large")
    
    with c_main:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Probabilidad del Modelo")
        st.pyplot(draw_custom_gauge(prob, "Riesgo Calculado"), use_container_width=True)
        st.markdown(f"""
            <p style="font-size:0.9rem; color:#666; text-align:center;">
                El modelo estima una probabilidad del <strong>{prob*100:.1f}%</strong> basándose en 6 biomarcadores clave.
                Umbral de corte clínico: 0.27
            </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_pop:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Comparativa Poblacional")
        st.write("Posición del paciente respecto a la cohorte de referencia:")
        
        # 1. VISUALIZADOR DE GLUCOSA (Population Range)
        st.markdown("**Nivel de Glucosa** (Percentil)")
        
        # Lógica visual simple con HTML para barras de rango
        marker_pos = min(100, max(0, (glucose - 70) / (200 - 70) * 100))
        st.markdown(f"""
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#888;">
                <span>Hipoglucemia</span><span>Normal</span><span>Prediabetes</span><span>Diabetes</span>
            </div>
            <div style="width:100%; height:12px; background: linear-gradient(90deg, #4caf50 30%, #ffeb3b 60%, {CEMP_PINK} 100%); border-radius:6px; position:relative; margin-bottom:20px;">
                <div style="position:absolute; left:{marker_pos}%; top:-5px; width:4px; height:22px; background:#333; border:1px solid white;"></div>
                <div style="position:absolute; left:{marker_pos}%; top:-25px; transform:translateX(-50%); font-weight:bold; color:{TEXT_DARK};">{glucose} mg/dL</div>
            </div>
        """, unsafe_allow_html=True)

        # 2. VISUALIZADOR DE BMI (Categorías)
        st.markdown("**Índice de Masa Corporal (BMI)**")
        bmi_pos = min(100, max(0, (bmi - 15) / (40 - 15) * 100))
        st.markdown(f"""
             <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#888;">
                <span>Bajo</span><span>Normal</span><span>Sobrepeso</span><span>Obesidad</span>
            </div>
            <div style="width:100%; height:12px; background: linear-gradient(90deg, #2196f3 18%, #4caf50 35%, #ffeb3b 50%, {CEMP_PINK} 80%); border-radius:6px; position:relative;">
                <div style="position:absolute; left:{bmi_pos}%; top:-5px; width:4px; height:22px; background:#333; border:1px solid white;"></div>
                 <div style="position:absolute; left:{bmi_pos}%; top:-25px; transform:translateX(-50%); font-weight:bold; color:{TEXT_DARK};">{bmi:.1f}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # SECCIÓN INFERIOR: TARJETAS DE DETALLE
    c_det1, c_det2, c_det3 = st.columns(3)
    with c_det1:
        st.info("💡 **HOMA-IR:** Indicador de resistencia a la insulina. Valores > 2.5 sugieren resistencia.")
    with c_det2:
        st.warning("⚖️ **Metabolismo:** El BMI elevado es el factor de riesgo modificable más alto en este perfil.")
    with c_det3:
        st.success("✅ **Edad:** Paciente joven (<50), lo que mejora el pronóstico de reversión.")

# --- TAB 2: EXPLICABILIDAD (SHAP MEJORADO) ---
with tab2:
    st.subheader("Desglose de Factores de Influencia")
    col_shap_text, col_shap_plot = st.columns([1, 2])
    
    with col_shap_text:
        st.markdown("""
        Este gráfico explica **por qué** el modelo dio este resultado.
        * **Barras Rosas (Derecha):** Aumentan el riesgo.
        * **Barras Grises (Izquierda):** Protegen contra la diabetes.
        """)
        if glucose > 140:
            st.markdown(f"<div class='rec-box'>⚠️ La **Glucosa ({glucose})** es el factor dominante impulsando el riesgo.</div>", unsafe_allow_html=True)
        if bmi > 30:
            st.markdown(f"<div class='rec-box'>⚠️ El **BMI ({bmi})** está contribuyendo significativamente (+15% prob).</div>", unsafe_allow_html=True)

    with col_shap_plot:
        # Mock SHAP Plot Clean
        features = ["Glucosa", "BMI", "Edad", "Insulina", "Genética"]
        vals = [(glucose-100)/100, (bmi-25)/50, -0.1, 0.05, 0.02]
        
        fig_s, ax_s = plt.subplots(figsize=(8, 4))
        fig_s.patch.set_facecolor('none')
        ax_s.set_facecolor('none')
        
        y = np.arange(len(features))
        colors = [CEMP_PINK if x > 0 else "#999" for x in vals]
        
        ax_s.barh(y, vals, color=colors, height=0.6)
        ax_s.set_yticks(y)
        ax_s.set_yticklabels(features, fontsize=12, color=TEXT_DARK)
        ax_s.axvline(0, color='#ddd')
        ax_s.spines['top'].set_visible(False)
        ax_s.spines['right'].set_visible(False)
        ax_s.spines['bottom'].set_visible(False)
        ax_s.spines['left'].set_visible(False)
        ax_s.tick_params(axis='x', colors='#888')
        
        st.pyplot(fig_s)

# --- TAB 3: PLAN DE ACCIÓN (NUEVO) ---
with tab3:
    st.subheader("🤖 Recomendaciones Clínicas Generadas por IA")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("#### 🥗 Nutrición y Estilo de Vida")
        if glucose > 120:
            st.markdown("• **Restricción de Carbohidratos:** Limitar índice glucémico < 55.")
            st.markdown("• **Ayuno Intermitente:** Considerar protocolo 16:8 bajo supervisión.")
        else:
            st.markdown("• **Dieta Mediterránea:** Mantener consumo alto de grasas saludables.")
            
        if bmi > 25:
            st.markdown("• **Déficit Calórico:** Objetivo de reducción de peso del 5-10%.")
            st.markdown("• **Actividad Física:** 150 min/semana de ejercicio moderado.")
            
    with col_rec2:
        st.markdown("#### 💊 Protocolo Médico Sugerido")
        if is_high:
            st.markdown("1. **Prueba Confirmatoria:** Solicitar HbA1c y Péptido C.")
            st.markdown("2. **Monitorización:** Control de glucosa capilar en ayunas durante 14 días.")
            st.markdown("3. **Farmacología:** Evaluar inicio de Metformina si HbA1c > 6.0%.")
        else:
            st.markdown("1. **Screening Anual:** Repetir panel metabólico en 12 meses.")
            st.markdown("2. **Educación:** Reforzar hábitos preventivos.")

    st.markdown("---")
    st.button("📥 Descargar Informe Completo (PDF)", type="primary")
