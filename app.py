import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# --- 1. CONFIGURACIÓN ---
st.set_page_config(
    page_title="CEMP AI", 
    page_icon="🩺", 
    layout="wide"
)

# --- 2. COLORES ---
CEMP_PINK = "#E97F87"
CEMP_DARK = "#2C3E50"
GOOD_TEAL = "#4DB6AC"
# Color gris para el borde del tirador del slider
SLIDER_GRAY = "#BDC3C7"
RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

# --- 3. CSS (ESTILOS AVANZADOS) ---
st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* CONTENEDOR PRINCIPAL */
    .block-container {{
        max-width: 1250px; 
        padding-top: 2rem;
        padding-bottom: 3rem;
        margin: 0 auto;
    }}
    
    /* LOGO */
    .cemp-logo {{ font-family: 'Helvetica', sans-serif; font-weight: 800; font-size: 1.8rem; color: {CEMP_DARK}; margin:0; }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}

    /* ==================================================================
       ESTILO SLIDER UMBRAL (FONDO ROSA SUTIL + LÍNEA BLANCA)
       ================================================================== */
    .stMain .stSlider {{
        background-color: rgba(233, 127, 135, 0.1) !important;
        padding: 20px 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        border: none !important;
    }}
    .stMain .stSlider label p {{
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        color: #999 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .stMain .stSlider [data-testid="stMarkdownContainer"] p {{
         color: {CEMP_DARK} !important; 
         font-weight: 800 !important;
         font-size: 1rem !important;
    }}
    .stMain .stSlider [role="slider"] {{
        background-color: white !important;
        border: 2px solid {SLIDER_GRAY} !important; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .stMain .stSlider > div > div > div > div {{
        background: white !important;
        color: white !important;
    }}
    .stMain .stSlider > div > div > div > div > div {{
         background-color: rgba(255, 255, 255, 0.5) !important;
    }}

    /* ==================================================================
       ESTILO PARA LA BARRA LATERAL (INPUTS SINCRONIZADOS)
       ================================================================== */
    /* Ajustamos el number_input para que se vea limpio al lado del slider */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
        padding: 0px 5px;
        font-size: 0.9rem;
        text-align: center;
        color: {CEMP_DARK};
        font-weight: bold;
    }}

    /* TARJETAS ESTÁNDAR */
    .card {{
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    
    .card-header {{
        color: #999;
        font-size: 0.75rem;
        font-weight: bold;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 15px;
        display: block;
    }}

    /* KPI SIDEBAR */
    .kpi-box {{
        background: white; border-left: 4px solid {CEMP_PINK};
        padding: 12px; border-radius: 6px; margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }}
    
    /* BARRAS DE PROGRESO */
    .bar-container {{
        position: relative; width: 100%; margin-top: 15px; margin-bottom: 25px;
    }}
    .bar-bg {{ background: #F0F2F5; height: 10px; border-radius: 5px; width: 100%; overflow: hidden; }}
    .bar-fill {{ height: 100%; width: 100%; background: {RISK_GRADIENT}; border-radius: 5px; opacity: 0.9; }}
    .bar-marker {{ 
        position: absolute; top: -5px; width: 4px; height: 20px; 
        background: {CEMP_DARK}; border: 1px solid white; border-radius: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2); z-index: 10; transition: left 0.3s ease;
    }}
    .bar-txt {{ 
        position: absolute; top: -28px; transform: translateX(-50%); 
        font-size: 0.8rem; font-weight: bold; color: {CEMP_DARK}; 
        background: white; padding: 2px 6px; border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }}
    .legend-row {{ display: flex; justify-content: space-between; font-size: 0.7rem; color: #BBB; margin-top: -5px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
    
    </style>
""", unsafe_allow_html=True)

# --- 4. HELPER (IMÁGENES) ---
def fig_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%; object-fit:contain;">'

# --- 5. MODELO MOCK ---
if 'model' not in st.session_state:
    class MockModel:
        def predict_proba(self, X):
            score = (X[0]*0.5) + (X[1]*0.4) + (X[3]*0.1) 
            prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

# --- 6. FUNCION HELPER PARA INPUTS SINCRONIZADOS ---
def input_biomarker(label, min_val, max_val, default_val, key):
    """Crea un slider y un number_input sincronizados."""
    st.markdown(f"**{label}**")
    c1, c2 = st.columns([3, 1]) # Columna ancha para slider, estrecha para cajita
    
    # Inicializar estado si no existe
    if key not in st.session_state:
        st.session_state[key] = default_val

    # Callbacks para sincronizar
    def update_from_slider():
        st.session_state[key] = st.session_state[f"{key}_slider"]
    def update_from_box():
        st.session_state[key] = st.session_state[f"{key}_box"]

    with c1:
        st.slider(
            label="", # Ocultamos label porque ya pusimos el título arriba
            min_value=min_val, max_value=max_val, 
            key=f"{key}_slider", 
            value=st.session_state[key], 
            on_change=update_from_slider,
            label_visibility="collapsed"
        )
    with c2:
        st.number_input(
            label="", 
            min_value=min_val, max_value=max_val, 
            key=f"{key}_box", 
            value=st.session_state[key], 
            on_change=update_from_box,
            label_visibility="collapsed"
        )
    return st.session_state[key]

# --- 7. BARRA LATERAL (NUEVA VERSIÓN) ---
with st.sidebar:
    st.markdown('<div class="cemp-logo">CEMP<span>.</span>AI</div>', unsafe_allow_html=True)
    st.caption("CLINICAL DECISION SUPPORT SYSTEM")
    st.write("")
    
    st.markdown("### 🧬 Biomarcadores")
    
    # Usamos la nueva función para inputs cómodos
    glucose = input_biomarker("Glucosa (mg/dL)", 50, 250, 120, "gluc")
    bmi = input_biomarker("BMI (kg/m²)", 15.0, 50.0, 28.5, "bmi")
    insulin = input_biomarker("Insulina (mu U/ml)", 0, 600, 100, "ins")
    age = input_biomarker("Edad (años)", 18, 90, 45, "age")
    
    st.write("")
    with st.expander("Factores Secundarios"):
        pregnancies = st.slider("Embarazos", 0, 15, 1)
        dpf = st.slider("Función Pedigrí", 0.0, 2.5, 0.5)

    st.markdown("---")
    
    # KPIs
    homa = glucose * insulin / 405
    c1, c2 = st.columns(2)
    with c1: st.markdown(f'<div class="kpi-box"><div style="font-size:1.4rem; font-weight:bold; color:{CEMP_DARK}">{homa:.1f}</div><div style="font-size:0.7rem; color:#888; font-weight:600;">HOMA-IR</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-box"><div style="font-size:1.4rem; font-weight:bold; color:{CEMP_DARK}">{bmi:.1f}</div><div style="font-size:0.7rem; color:#888; font-weight:600;">BMI</div></div>', unsafe_allow_html=True)
    

# --- 8. INTERFAZ PRINCIPAL ---

st.markdown(f"<h1 style='color:{CEMP_DARK}; margin-bottom: 20px; font-size: 2.2rem;'>Perfil de Riesgo Metabólico</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Panel General", "Factores (SHAP)", "Protocolo"])

# --- PESTAÑA 1 ---
with tab1:
    st.write("")
    
    # UMBRAL
    threshold = st.slider("Umbral de Decisión Clínica (Ajuste de Sensibilidad)", 0.0, 1.0, 0.31, 0.01)

    # LÓGICA
    input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
    prob = st.session_state.model.predict_proba(input_data)[0][1]
    is_high = prob > threshold # Aquí es donde el umbral decide si es ALTO o BAJO
    
    risk_color = CEMP_PINK if is_high else GOOD_TEAL
    risk_label = "ALTO RIESGO" if is_high else "BAJO RIESGO"
    risk_icon = "🔴" if is_high else "🟢"
    risk_bg = "#FFF5F5" if is_high else "#F0FDF4"
    risk_border = CEMP_PINK if is_high else GOOD_TEAL
    
    alerts = []
    if glucose > 120: alerts.append("Hiperglucemia")
    if bmi > 30: alerts.append("Obesidad")
    if homa > 2.5: alerts.append("Resistencia Insulina")
    insight_txt = " • ".join(alerts) if alerts else "Paciente estable"
    insight_bd = CEMP_PINK if alerts else GOOD_TEAL

    # LAYOUT
    c_left, c_right = st.columns([1.8, 1], gap="medium") 
    
    # === IZQUIERDA ===
    with c_left:
        # Ficha Paciente
        st.markdown(f"""<div class="card" style="flex-direction:row; align-items:center; justify-content:space-between;">
<div style="display:flex; align-items:center; gap:20px; flex-grow:1;">
<div style="background:rgba(233, 127, 135, 0.1); width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2rem; color:{CEMP_DARK};">👤</div>
<div>
<span class="card-header" style="margin-bottom:5px;">EXPEDIENTE MÉDICO</span>
<h2 style="margin:0; color:{CEMP_DARK}; font-size:1.6rem; line-height:1.2;">Paciente #8842-X</h2>
<div style="font-size:0.85rem; color:#666; margin-top:5px;">📅 Revisión: <b>14 Dic 2025</b></div>
</div>
</div>
<div style="background:{risk_bg}; border:1px solid {risk_border}; color:{risk_border}; font-weight:bold; font-size:0.9rem; padding:8px 16px; border-radius:30px;">
{risk_icon} {risk_label}
</div>
</div>""", unsafe_allow_html=True)

        # Contexto Poblacional
        g_pos = min(100, max(0, (glucose - 60) / 1.4))
        b_pos = min(100, max(0, (bmi - 18) / 0.22))
        
        st.markdown(f"""<div class="card">
<span class="card-header">CONTEXTO POBLACIONAL</span>
<div style="margin-top:15px;">
<div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">GLUCOSA BASAL <span style="font-weight:normal">({glucose} mg/dL)</span></div>
<div class="bar-container">
<div class="bar-bg"><div class="bar-fill"></div></div>
<div class="bar-marker" style="left: {g_pos}%;"></div>
<div class="bar-txt" style="left: {g_pos}%;">{glucose}</div>
</div>
<div class="legend-row">
<span>Hipoglucemia</span><span>Normal</span><span>Prediabetes</span><span>Diabetes</span>
</div>
</div>
<div style="margin-top:35px;">
<div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">ÍNDICE DE MASA CORPORAL <span style="font-weight:normal">({bmi})</span></div>
<div class="bar-container">
<div class="bar-bg"><div class="bar-fill"></div></div>
<div class="bar-marker" style="left: {b_pos}%;"></div>
<div class="bar-txt" style="left: {b_pos}%;">{bmi}</div>
</div>
<div class="legend-row">
<span>Sano</span><span>Sobrepeso</span><span>Obesidad G1</span><span>Obesidad G2</span>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # === DERECHA ===
    with c_right:
        # Hallazgos
        st.markdown(f"""<div class="card" style="border-left:5px solid {insight_bd}; justify-content:center;">
    <span class="card-header" style="color:{insight_bd}; margin-bottom:10px;">HALLAZGOS CLAVE</span>
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h3 style="margin:0; color:{CEMP_DARK}; font-size:1.1rem; line-height:1.4;">{insight_txt}</h3>
        <div style="font-size:1.8rem;">{'⚠️' if alerts else '✅'}</div>
    </div>
</div>""", unsafe_allow_html=True)
        
        # Probabilidad IA
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.pie([prob, 1-prob], colors=[risk_color, '#F4F6F9'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15, edgecolor='none'))
        chart_html = fig_to_html(fig)
        plt.close(fig)

        st.markdown(f"""<div class="card" style="text-align:center; padding: 40px 20px;">
    <span class="card-header" style="margin-bottom:20px;">PROBABILIDAD IA</span>
    <div style="position:relative; display:inline-block; margin: auto;">
        {chart_html}
        <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-size:2.5rem; font-weight:800; color:{CEMP_DARK}; letter-spacing:-1px;">
            {prob*100:.1f}%
        </div>
    </div>
    <div style="font-size:0.8rem; color:#888; margin-top:20px;">Confianza: <strong>Alta</strong> <br> Umbral: {threshold}</div>
</div>""", unsafe_allow_html=True)

# --- TAB 2: SHAP ---
with tab2:
    st.write("")
    features = ["Glucosa", "BMI", "Edad", "Insulina"]
    vals = [(glucose-100)/100, (bmi-25)/50, -0.1, 0.05]
    colors = [CEMP_PINK if x>0 else "#BDC3C7" for x in vals]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.barh(features, vals, color=colors, height=0.6)
    ax.axvline(0, color='#eee')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', colors='#999')
    ax.tick_params(axis='y', labelsize=10, labelcolor=CEMP_DARK)
    chart_html = fig_to_html(fig)
    plt.close(fig)

    st.markdown(f"""<div class="card">
<h3 style="color:{CEMP_DARK}; font-size:1.2rem; margin-bottom:5px;">Factores de Riesgo (SHAP)</h3>
<span class="card-header" style="margin-bottom:20px;">EXPLICABILIDAD DEL MODELO</span>
{chart_html}
</div>""", unsafe_allow_html=True)

# --- TAB 3: PROTOCOLO ---
with tab3:
    st.write("")
    st.info("💡 Módulo de recomendaciones clínicas y generación de informes.")
