import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import date
import time

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="DIABETES.NME | TFM", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. GESTIÓN DE ESTADO ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'patient' not in st.session_state: st.session_state.patient = {'id': '', 'name': '', 'date': date.today()}
if 'threshold' not in st.session_state: st.session_state.threshold = 0.27
if 'model' not in st.session_state:
    class MockModel:
        def predict_proba(self, X):
            score = (X[0]*0.5) + (X[1]*0.6) + (X[3]*0.1) 
            prob = 1 / (1 + np.exp(-(score - 110) / 20)) 
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

# --- 3. COLORES Y ESTILOS ---
CEMP_PINK = "#E97F87"
CEMP_DARK = "#2C3E50"
GOOD_TEAL = "#4DB6AC"
SLIDER_GRAY = "#BDC3C7"
OPTIMAL_GREEN = "#8BC34A"
NOTE_GRAY_BG = "#F8F9FA"
NOTE_GRAY_TEXT = "#6C757D"

RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"
BMI_GRADIENT = "linear-gradient(90deg, #81D4FA 0%, #4DB6AC 25%, #FFF176 40%, #FFB74D 55%, #E97F87 70%, #880E4F 100%)"
GLUCOSE_GRADIENT = "linear-gradient(90deg, #4DB6AC 0%, #4DB6AC 28%, #FFF176 32%, #FFB74D 48%, #E97F87 52%, #880E4F 100%)"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1250px; }}

    /* LOGO SIDEBAR */
    .cemp-logo {{ font-family: 'Helvetica', sans-serif; font-weight: 900; color: {CEMP_DARK}; display: flex; align-items: center; }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}

    /* === PORTADA (LANDING) === */
    .landing-wrapper {{
        background: linear-gradient(145deg, #FFFFFF 0%, #FFF5F6 100%);
        padding: 50px 40px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(233, 127, 135, 0.15);
        box-shadow: 0 20px 40px rgba(233, 127, 135, 0.08);
        margin-top: 30px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }}
    
    .cemp-badge {{
        display: inline-block;
        background-color: {CEMP_DARK};
        color: #FFF;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 20px;
        text-transform: uppercase;
    }}

    /* TÍTULOS AJUSTADOS DE TAMAÑO */
    .landing-institution {{
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
        font-size: 1rem; /* Reducido */
        color: {CEMP_DARK};
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}

    .landing-title-text {{
        font-family: 'Helvetica', sans-serif;
        font-weight: 900;
        font-size: 3.5rem; /* Reducido para que quepa mejor */
        color: {CEMP_DARK};
        line-height: 1.1;
        letter-spacing: -1.5px;
        margin-bottom: 20px;
        margin-top: 5px;
    }}
    
    .landing-pink {{ color: {CEMP_PINK}; }}
    .landing-gray {{ color: {SLIDER_GRAY}; }}
    
    .landing-hero-text {{
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem; /* Ajustado */
        font-weight: 700;
        color: {CEMP_DARK};
        margin-bottom: 20px;
        line-height: 1.4;
    }}
    
    .landing-description {{
        font-size: 1rem;
        color: #666;
        line-height: 1.6;
        max-width: 700px;
        margin: 0 auto 35px auto;
    }}

    .disclaimer-box {{
        background-color: #F8F9FA;
        border-left: 4px solid {CEMP_PINK};
        padding: 20px;
        margin: 0 auto 30px auto;
        text-align: left;
        font-size: 0.85rem;
        color: #555;
        border-radius: 8px;
        max-width: 750px;
        line-height: 1.5;
    }}
    
    /* BOTONES */
    div.stButton > button:first-child {{
        background-color: {CEMP_PINK}; color: white; font-weight: 800; font-size: 1.1rem;
        padding: 1rem 3rem; border-radius: 16px; border: none;
        width: auto; min-width: 280px; text-transform: uppercase; letter-spacing: 1px; transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(233, 127, 135, 0.4);
    }}
    div.stButton > button:first-child:hover {{
        background-color: #D66E76; transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(233, 127, 135, 0.5);
    }}
    
    /* BOTÓN SECUNDARIO */
    .secondary-btn button {{
        background-color: transparent !important;
        border: 2px solid {CEMP_DARK} !important;
        color: {CEMP_DARK} !important;
        box-shadow: none !important; padding: 0.6rem 2rem !important; font-size: 0.9rem !important;
    }}
    .secondary-btn button:hover {{
        background-color: {CEMP_DARK} !important; color: white !important; transform: translateY(-2px);
    }}

    /* SIDEBAR */
    .stSlider {{ padding-top: 0px !important; padding-bottom: 10px !important; }}
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
        padding: 0px 5px; font-size: 0.9rem; text-align: center; color: {CEMP_DARK}; font-weight: 800;
        border-radius: 8px; background-color: white; border: 1px solid #ddd;
    }}
    .calc-box {{
        background-color: #F8F9FA; border-radius: 8px; padding: 12px 15px; border: 1px solid #EEE;
        margin-top: 5px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        border-left: 4px solid {CEMP_PINK};
    }}
    .calc-label {{ font-size: 0.75rem; color: #888; font-weight: 600; text-transform: uppercase; }}
    .calc-value {{ font-size: 1rem; color: {CEMP_DARK}; font-weight: 800; }}

    /* TARJETAS */
    .card {{
        background-color: white; border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 15px; display: flex; flex-direction: column; justify-content: center; min-height: 300px; 
    }}
    .card-auto {{ min-height: auto !important; height: 100%; }}
    .card-header {{ color: #999; font-size: 0.75rem; font-weight: bold; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 15px; display: flex; align-items: center; }}
    
    /* GRÁFICOS */
    .bar-container {{ position: relative; width: 100%; margin-top: 20px; margin-bottom: 30px; }}
    .bar-bg {{ background: #F0F2F5; height: 12px; border-radius: 6px; width: 100%; overflow: hidden; }}
    .bar-fill {{ height: 100%; width: 100%; background: {RISK_GRADIENT}; border-radius: 6px; }}
    .bar-fill-bmi {{ height: 100%; width: 100%; background: {BMI_GRADIENT}; border-radius: 6px; }}
    .bar-fill-glucose {{ height: 100%; width: 100%; background: {GLUCOSE_GRADIENT}; border-radius: 6px; }}
    .bar-marker {{ position: absolute; top: -6px; width: 4px; height: 24px; background: {CEMP_DARK}; border: 1px solid white; border-radius: 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index:10; transition: left 0.3s; }}
    .bar-txt {{ position: absolute; top: -30px; transform: translateX(-50%); font-size: 0.85rem; font-weight: bold; color: {CEMP_DARK}; background: white; padding: 2px 8px; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    .legend-container {{ position: relative; width: 100%; height: 20px; margin-top: 8px; }}
    .legend-label {{ position: absolute; transform: translateX(-50%); font-size: 0.7rem; color: #888; font-weight: 600; white-space: nowrap; }}
    
    </style>
""", unsafe_allow_html=True)

# --- 4. FUNCIONES AUXILIARES ---
def fig_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%; object-fit:contain;">'

def get_help_icon(description):
    return f"""<span style="display:inline-block; width:16px; height:16px; line-height:16px; text-align:center; border-radius:50%; background:#E0E0E0; color:#777; font-size:0.7rem; font-weight:bold; cursor:help; margin-left:6px; position:relative; top:-1px;" title="{description}">?</span>"""

def generate_report(data_dict, prob, risk_label, alerts):
    return f"""==================================================\nINFORME TFM - DIABETES.NME\n==================================================\nFecha: {date.today().strftime("%d/%m/%Y")}\nID Paciente: {data_dict.get('id')}\n--------------------------------------------------\nRESULTADOS:\nRiesgo: {risk_label} ({prob*100:.1f}%)\nHallazgos: {', '.join(alerts)}\n=================================================="""

# Función de input sincronizado
def input_biomarker(label_text, min_val, max_val, default_val, key, help_text=""):
    label_html = f"**{label_text}**"
    if help_text: label_html += get_help_icon(help_text)
    st.markdown(label_html, unsafe_allow_html=True)
    c1, c2 = st.columns([2.5, 1], gap="small")
    input_type = type(default_val)
    min_val, max_val = input_type(min_val), input_type(max_val)
    step = 0.1 if input_type == float else 1
    if key not in st.session_state: st.session_state[key] = default_val
    def update_from_slider():
        st.session_state[key] = st.session_state[f"{key}_slider"]
        st.session_state[f"{key}_input"] = st.session_state[f"{key}_slider"] 
    def update_from_input():
        val = st.session_state[f"{key}_input"]
        val = max(min_val, min(val, max_val))
        st.session_state[key] = val
        st.session_state[f"{key}_slider"] = val 
    with c1: st.slider(label="", min_value=min_val, max_value=max_val, step=step, key=f"{key}_slider", value=st.session_state[key], on_change=update_from_slider, label_visibility="collapsed")
    with c2: st.number_input(label="", min_value=min_val, max_value=max_val, step=step, key=f"{key}_input", value=st.session_state[key], on_change=update_from_input, label_visibility="collapsed")
    return st.session_state[key]

# ==========================================
# NAVEGACIÓN Y PÁGINAS
# ==========================================

# === PASO 1: PORTADA (LANDING) ===
if st.session_state.step == 1:
    st.markdown("""<style>[data-testid="stSidebar"] {display: none;}</style>""", unsafe_allow_html=True)
    st.write("")
    
    with st.container():
        st.markdown(f"""<div class="landing-wrapper">
<div class="cemp-badge">TFM • MÁSTER EN INTELIGENCIA ARTIFICIAL APLICADA A LA SALUD</div>
<div class="landing-institution">CENTRO EUROPEO DE MÁSTERES Y POSGRADOS</div>
<div class="landing-title-text">
D<span class="landing-pink">IA</span>BETES<span class="landing-gray">.</span><span class="landing-pink">NME</span>
</div>
<div class="landing-hero-text">
Prototipo de CDSS para el diagnóstico temprano de diabetes
</div>
<p class="landing-description">
Este proyecto explora el potencial de integrar modelos predictivos avanzados en el flujo de trabajo clínico, visualizando un futuro donde la IA actúa como un potente aliado en la detección temprana y prevención de la diabetes tipo 2.
</p>
<div class="disclaimer-box">
<strong>Aplicación desarrollada con fines exclusivamente educativos e investigativos como parte de un Trabajo de Fin de Máster.</strong><br><br>
⚠️ Esta herramienta NO es un dispositivo médico certificado. Los resultados son una simulación académica y NO deben utilizarse para el diagnóstico real, tratamiento o toma de decisiones clínicas.
</div>
</div>""", unsafe_allow_html=True)
        
        # --- BOTÓN CENTRADO ---
        st.markdown('<div style="display: flex; justify-content: center; margin-top: -30px; position: relative; z-index: 10;">', unsafe_allow_html=True)
        if st.button("INICIAR SIMULACIÓN  ➔", key="landing_btn"):
            st.session_state.step = 2
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# === PASO 2 Y 3: NECESITAN LA BARRA LATERAL ===
if st.session_state.step > 1:
    
    # --- RENDERIZADO DE LA BARRA LATERAL ---
    with st.sidebar:
        st.markdown(f'<div class="cemp-logo" style="font-size: 1.8rem;">D<span>IA</span>BETES<span style="color:{SLIDER_GRAY}">.</span><span>NME</span></div>', unsafe_allow_html=True)
        st.caption("CLINICAL DECISION SUPPORT SYSTEM | TFM")
        st.write("")
        st.markdown("**1. Parámetros Clínicos**")
        glucose = input_biomarker("Glucosa 2h (mg/dL)", 50, 350, 120, "gluc", "Concentración plasmática a las 2h de test de tolerancia oral.")
        insulin = input_biomarker("Insulina (µU/ml)", 0, 900, 100, "ins", "Insulina a las 2h de ingesta.")
        proxy_index = glucose * insulin
        st.markdown(f"""<div class="calc-box"><div style="display:flex; justify-content:space-between; align-items:center;"><span class="calc-label">Índice RI (Glucosa x Insulina)</span><span class="calc-value">{proxy_index:,.0f}</span></div></div>""", unsafe_allow_html=True)
        st.markdown("---") 
        st.markdown("**2. Antropometría**")
        weight = input_biomarker("Peso (kg)", 30.0, 250.0, 70.0, "weight", "Peso corporal actual.")
        height = input_biomarker("Altura (m)", 1.00, 2.20, 1.70, "height", "Altura en metros.")
        bmi = weight / (height * height)
        bmi_sq = bmi ** 2
        st.markdown(f"""<div class="calc-box"><div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span class="calc-label">BMI (kg/m²)</span><span class="calc-value">{bmi:.2f}</span></div><div style="display:flex; justify-content:space-between;"><span class="calc-label">BMI² (Non-Linear)</span><span class="calc-value">{bmi_sq:.2f}</span></div></div>""", unsafe_allow_html=True)
        st.markdown("---") 
        st.markdown("**3. Historia**")
        age = input_biomarker("Edad (años)", 18, 90, 45, "age")
        pregnancies = input_biomarker("Embarazos", 0, 20, 1, "preg", "Nº veces embarazada.") 
        st.markdown("---") 
        st.markdown("**4. Genética**")
        dpf = input_biomarker("Antecedentes (DPF)", 0.0, 2.5, 0.5, "dpf", "Estimación de predisposición genética.")
        if dpf <= 0.15: dpf_label, bar_color = "Carga familiar MUY BAJA", GOOD_TEAL
        elif dpf <= 0.40: dpf_label, bar_color = "Carga familiar BAJA", "#D4E157"
        elif dpf <= 0.80: dpf_label, bar_color = "Carga familiar MODERADA", "#FFB74D"
        elif dpf <= 1.20: dpf_label, bar_color = "Carga familiar ELEVADA", CEMP_PINK
        else: dpf_label, bar_color = "Carga familiar MUY ELEVADA", "#880E4F"
        st.markdown(f"""<div style="display:flex; justify-content:space-between; align-items:center; margin-top:-10px; margin-bottom:2px;"><span style="font-size:0.8rem; font-weight:bold; color:{bar_color};">{dpf_label}</span><span style="font-size:0.8rem; color:#666;">{dpf:.2f}</span></div><div style="width:100%; background-color:#F0F2F5; border-radius:4px; height:8px; margin-bottom:10px;"><div style="width:{min(100, (dpf/2.5)*100)}%; background-color:{bar_color}; height:8px; border-radius:4px; transition: width 0.3s ease, background-color 0.3s ease;"></div></div>""", unsafe_allow_html=True)
        st.caption("Valores basados en el estudio Pima Indians Diabetes.")

# === PASO 2: REGISTRO DE PACIENTE Y CONFIGURACIÓN ===
if st.session_state.step == 2:
    st.markdown(f"<h1 style='color:{CEMP_DARK};'>Registro para Simulación</h1>", unsafe_allow_html=True)
    st.write("Introduzca los datos administrativos y configure el umbral de sensibilidad del modelo antes de realizar la predicción.")
    with st.container():
        c_form1, c_form2 = st.columns(2)
        with c_form1: st.session_state.patient['id'] = st.text_input("ID Paciente / Historia Clínica", value=st.session_state.patient['id'], placeholder="Ej: 8842-X")
        with c_form2: st.session_state.patient['date'] = st.date_input("Fecha de Consulta", value=st.session_state.patient['date'])
        st.session_state.patient['name'] = st.text_input("Nombre Completo (Opcional)", value=st.session_state.patient['name'], placeholder="Nombre y Apellidos")
    st.write(""); st.markdown("---")
    
    st.subheader("⚙️ Configuración del Modelo (Random Forest)")
    c_thresh1, c_thresh2 = st.columns([1, 2])
    with c_thresh1:
        st.session_state.threshold = st.slider("Umbral de Decisión", 0.0, 1.0, st.session_state.threshold, 0.01)
        st.info(f"Umbral seleccionado: **{st.session_state.threshold}**")
    with c_thresh2:
        st.markdown(f"""<div style="font-size:0.9rem; color:#666; padding-top:10px;">Define el punto de corte probabilístico para clasificar un caso como positivo. <br>• <strong>0.27 (Recomendado):</strong> Optimiza F2-Score para este modelo (minimiza falsos negativos).<br>• <strong>0.50 (Estándar):</strong> Balance neutro entre sensibilidad y especificidad.</div>""", unsafe_allow_html=True)
    st.write(""); st.write("")
    
    col_back, col_pred, col_dummy = st.columns([1, 2, 1])
    with col_back:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("⬅ VOLVER A PORTADA"):
            st.session_state.step = 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_pred:
        if st.button("GENERAR PREDICCIÓN CLÍNICA  ➔"):
            if not st.session_state.patient['id']:
                st.error("⚠️ Por favor, introduzca al menos un ID de paciente para continuar.")
            else:
                with st.spinner("Procesando datos y ejecutando modelo..."):
                    time.sleep(1)
                    st.session_state.step = 3
                    st.rerun()

# === PASO 3: RESULTADOS (DASHBOARD ORIGINAL) ===
if st.session_state.step == 3:
    c_head, c_nav = st.columns([4, 1])
    with c_head: st.markdown(f"<h1 style='color:{CEMP_DARK}; margin-bottom: 0px;'>Resultados del Análisis</h1>", unsafe_allow_html=True)
    with c_nav:
        st.write("")
        if st.button("✏️ MODIFICAR DATOS"):
            st.session_state.step = 2
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["Panel General", "Factores (SHAP)", "Protocolo"])
    with tab1:
        st.write("")
        threshold = st.session_state.threshold
        input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
        prob = st.session_state.model.predict_proba(input_data)[0][1]
        is_high = prob > threshold 
        
        distancia_al_corte = abs(prob - threshold)
        if distancia_al_corte > 0.15: conf_text, conf_color = "ALTA", GOOD_TEAL
        elif distancia_al_corte > 0.05: conf_text, conf_color = "MEDIA", "#F39C12"
        else: conf_text, conf_color = "BAJA", CEMP_PINK
        conf_desc = "Fiabilidad basada en la distancia de la probabilidad al umbral seleccionado."

        risk_color = CEMP_PINK if is_high else GOOD_TEAL
        risk_label = "ALTO RIESGO" if is_high else "BAJO RIESGO"
        risk_icon = "🔴" if is_high else "🟢"
        risk_bg = "#FFF5F5" if is_high else "#F0FDF4"
        risk_border = CEMP_PINK if is_high else GOOD_TEAL
        
        alerts = []
        if glucose >= 200: alerts.append("Posible Diabetes")
        elif glucose >= 140: alerts.append("Posible Prediabetes")
        if bmi >= 40: alerts.append("Obesidad Mórbida (G3)")
        elif bmi >= 35: alerts.append("Obesidad G2")
        elif bmi >= 30: alerts.append("Obesidad G1")
        elif bmi >= 25: alerts.append("Sobrepeso")
        elif bmi < 18.5: alerts.append("Bajo Peso")
        if proxy_index > 19769.5: alerts.append("Resistencia Insulina")
        
        if not alerts: insight_txt, insight_bd, alert_icon = "Sin hallazgos significativos", GOOD_TEAL, "✅"
        else: insight_txt, insight_bd, alert_icon = " • ".join(alerts), CEMP_PINK, "⚠️"

        c_left, c_right = st.columns([1.8, 1], gap="medium") 
        with c_left:
            pat_name = st.session_state.patient['name'] if st.session_state.patient['name'] else "---"
            pat_id = st.session_state.patient['id']
            pat_date = st.session_state.patient['date'].strftime("%d %b %Y")
            st.markdown(f"""<div class="card card-auto" style="flex-direction:row; align-items:center; justify-content:space-between;"><div style="display:flex; align-items:center; gap:20px; flex-grow:1;"><div style="background:rgba(233, 127, 135, 0.1); width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2rem; color:{CEMP_DARK};">👤</div><div><span class="card-header" style="margin-bottom:5px;">EXPEDIENTE MÉDICO</span><h2 style="margin:0; color:{CEMP_DARK}; font-size:1.6rem; line-height:1.2;">Paciente {pat_name}</h2><div style="font-size:0.85rem; color:#666; margin-top:5px;">ID: <b>{pat_id}</b> &nbsp;|&nbsp; Fecha: <b>{pat_date}</b></div></div></div><div style="display:flex; flex-direction:column; align-items:center; gap:8px;"><div style="background:{risk_bg}; border:1px solid {risk_border}; color:{risk_border}; font-weight:bold; font-size:0.9rem; padding:8px 16px; border-radius:30px;">{risk_icon} {risk_label}</div><div style="background:#F8F9FA; border-radius:8px; padding: 4px 10px; border:1px solid #EEE;" title="{conf_desc}"><span style="font-size:0.7rem; color:#999; font-weight:600;">FIABILIDAD: </span><span style="font-size:0.75rem; color:{conf_color}; font-weight:800;">{conf_text}</span></div></div></div>""", unsafe_allow_html=True)
            g_pos = min(100, max(0, (glucose - 50) / 3.0)) 
            b_pos = min(100, max(0, (bmi - 10) * 2.5)) 
            st.markdown(f"""<div class="card"><span class="card-header">CONTEXTO POBLACIONAL</span><div style="margin-top:15px;"><div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">GLUCOSA 2H (TEST TOLERANCIA) <span style="font-weight:normal">({glucose} mg/dL)</span></div><div class="bar-container"><div class="bar-bg"><div class="bar-fill-glucose"></div></div><div class="bar-marker" style="left: {g_pos}%;"></div><div class="bar-txt" style="left: {g_pos}%;">{glucose}</div></div><div class="legend-container"><span class="legend-label" style="left: 15%;">Normal (&lt;140)</span><span class="legend-label" style="left: 40%;">Intolerancia (140-199)</span><span class="legend-label" style="left: 75%;">Diabetes (&gt;200)</span></div></div><div style="margin-top:35px;"><div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">ÍNDICE DE MASA CORPORAL <span style="font-weight:normal">({bmi:.1f})</span></div><div class="bar-container"><div class="bar-bg"><div class="bar-fill-bmi"></div></div><div class="bar-marker" style="left: {b_pos}%;"></div><div class="bar-txt" style="left: {b_pos}%;">{bmi:.1f}</div></div><div class="legend-container"><span class="legend-label" style="left: 10%;">Bajo</span><span class="legend-label" style="left: 29%;">Normal</span><span class="legend-label" style="left: 43%;">Sobrepeso</span><span class="legend-label" style="left: 56%;">Ob. G1</span><span class="legend-label" style="left: 68%;">Ob. G2</span><span class="legend-label" style="left: 87%;">Ob. G3</span></div></div></div>""", unsafe_allow_html=True)
        with c_right:
            st.markdown(f"""<div class="card card-auto" style="border-left:5px solid {insight_bd}; justify-content:center;"><span class="card-header" style="color:{insight_bd}; margin-bottom:10px;">HALLAZGOS CLAVE</span><div style="display:flex; justify-content:space-between; align-items:center;"><h3 style="margin:0; color:{CEMP_DARK}; font-size:1.1rem; line-height:1.4;">{insight_txt}</h3><div style="font-size:1.8rem;">{alert_icon}</div></div></div>""", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(3.2, 3.2)); fig.patch.set_facecolor('none'); ax.set_facecolor('none')
            ax.pie([prob, 1-prob], colors=[risk_color, '#F4F6F9'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15, edgecolor='none'))
            threshold_angle = 90 - (threshold * 360); theta_rad = np.deg2rad(threshold_angle)
            ax.plot([0.85 * np.cos(theta_rad), 1.15 * np.cos(theta_rad)], [0.85 * np.sin(theta_rad), 1.15 * np.sin(theta_rad)], color=CEMP_DARK, linestyle='--', linewidth=2)
            chart_html = fig_to_html(fig); plt.close(fig)
            prob_help = get_help_icon("Probabilidad calculada por el modelo de IA.")
            st.markdown(f"""<div class="card" style="text-align:center; justify-content: center;"><span class="card-header" style="justify-content:center; margin-bottom:15px;">PROBABILIDAD IA{prob_help}</span><div style="position:relative; display:inline-block; margin: auto;">{chart_html}<div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-size:2.5rem; font-weight:800; color:{CEMP_DARK}; letter-spacing:-1px;">{prob*100:.1f}%</div></div><div style="margin-top: 8px; font-size: 0.65rem; color: #999; display: flex; align-items: center; justify-content: center; gap: 5px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase;"><span style="display: inline-block; width: 15px; border-top: 2px dashed {CEMP_DARK};"></span><span>Umbral de decisión ({threshold:.2f})</span></div></div>""", unsafe_allow_html=True)

    with tab2:
        st.write("")
        # (Gráfico SHAP estático como placeholder)
        features = ["Glucosa", "BMI", "Edad", "Insulina"]; vals = [(glucose-100)/100, (bmi-25)/50, -0.1, 0.05]
        colors = [CEMP_PINK if x>0 else "#BDC3C7" for x in vals]
        fig, ax = plt.subplots(figsize=(8, 4)); fig.patch.set_facecolor('none'); ax.set_facecolor('none')
        ax.barh(features, vals, color=colors, height=0.6); ax.axvline(0, color='#eee'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False); ax.tick_params(axis='x', colors='#999'); ax.tick_params(axis='y', labelsize=10, labelcolor=CEMP_DARK)
        chart_html = fig_to_html(fig); plt.close(fig)
        st.markdown(f"""<div class="card"><h3 style="color:{CEMP_DARK}; font-size:1.2rem; margin-bottom:5px;">Factores de Riesgo (SHAP)</h3><span class="card-header" style="margin-bottom:20px;">EXPLICABILIDAD DEL MODELO</span>{chart_html}</div>""", unsafe_allow_html=True)

    with tab3:
        st.write(""); st.info("💡 Módulo de recomendaciones clínicas y protocolos de actuación.")
    
    st.markdown("---")
    c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1])
    with c_btn1:
        report_text = generate_report(st.session_state.patient, prob, risk_label, alerts)
        st.download_button(label="📥 DESCARGAR INFORME CLÍNICO", data=report_text, file_name=f"Informe_{st.session_state.patient['id']}.txt", mime="text/plain")
    with c_btn3:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("🔄 NUEVA SIMULACIÓN COMPLETA"):
            st.session_state.step = 1; st.session_state.patient = {'id': '', 'name': '', 'date': date.today()}; st.session_state.threshold = 0.27
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
