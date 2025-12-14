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
RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

# --- 3. CSS (ESTILOS AVANZADOS) ---
st.markdown(f"""
    <style>
    /* Ocultar elementos base */
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
    
    /* === ESTILO ESPECIAL PARA EL SLIDER DEL UMBRAL (CAJA ROSA) === */
    /* Apuntamos solo a los sliders que están en el panel principal (no sidebar) */
    section.main div[data-testid="stSlider"] {{
        background-color: {CEMP_PINK};
        padding: 20px 25px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(233, 127, 135, 0.3);
        margin-bottom: 25px;
    }}
    /* Color del texto del label a BLANCO */
    section.main div[data-testid="stSlider"] label p {{
        color: white !important;
        font-weight: bold;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px;
    }}
    /* Color de los numeritos min/max a BLANCO */
    section.main div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
        opacity: 0.9;
    }}
    /* Intentar forzar color blanco en elementos del slider (depende del navegador) */
    section.main div[data-testid="stSlider"] div[role="slider"] {{
        background-color: white !important;
    }}
    
    /* TARJETAS (CARD) */
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
        justify-content: space-between;
    }}
    
    /* HEADER UNIFICADO DE LAS TARJETAS */
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
    
    /* BARRAS DE PROGRESO (Contexto) */
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
    
    /* LEYENDA */
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

# --- 6. BARRA LATERAL ---
with st.sidebar:
    st.markdown('<div class="cemp-logo">CEMP<span>.</span>AI</div>', unsafe_allow_html=True)
    st.caption("CLINICAL DECISION SUPPORT SYSTEM")
    st.write("")
    
    st.markdown("### 🧬 Biomarcadores")
    glucose = st.slider("Glucosa (mg/dL)", 50, 250, 120)
    bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 28.5)
    insulin = st.slider("Insulina (mu U/ml)", 0, 600, 100)
    age = st.slider("Edad (años)", 18, 90, 45)
    
    with st.expander("Factores Secundarios"):
        pregnancies = st.slider("Embarazos", 0, 15, 1)
        dpf = st.slider("Función Pedigrí", 0.0, 2.5, 0.5)

    st.markdown("---")
    
    # KPIs Rápidos
    homa = glucose * insulin / 405
    c1, c2 = st.columns(2)
    # Sin espacios al inicio del HTML para evitar errores
    with c1: st.markdown(f'<div class="kpi-box"><div style="font-size:1.4rem; font-weight:bold; color:{CEMP_DARK}">{homa:.1f}</div><div style="font-size:0.7rem; color:#888; font-weight:600;">HOMA-IR</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-box"><div style="font-size:1.4rem; font-weight:bold; color:{CEMP_DARK}">{bmi:.1f}</div><div style="font-size:0.7rem; color:#888; font-weight:600;">BMI</div></div>', unsafe_allow_html=True)
    

# --- 7. INTERFAZ PRINCIPAL ---

# Título
st.markdown(f"<h1 style='color:{CEMP_DARK}; margin-bottom: 20px; font-size: 2.2rem;'>Perfil de Riesgo Metabólico</h1>", unsafe_allow_html=True)

# Pestañas
tab1, tab2, tab3 = st.tabs(["Panel General", "Factores (SHAP)", "Protocolo"])

# --- PESTAÑA 1: DASHBOARD ---
with tab1:
    st.write("")
    
    # === 1. UMBRAL DE DECISIÓN (CAJA ROSA) ===
    # Este slider heredará automáticamente el estilo CSS definido arriba (Fondo rosa, letras blancas)
    threshold = st.slider("UMBRAL DE DECISIÓN CLÍNICA (SENSITIVITY ADJUSTMENT)", 0.0, 1.0, 0.27, 0.01)

    # --- LÓGICA DE CÁLCULO (Se ejecuta después del slider) ---
    input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
    prob = st.session_state.model.predict_proba(input_data)[0][1]
    is_high = prob > threshold # Usamos el umbral dinámico
    
    # Colores dinámicos
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

    # === LAYOUT: 2 COLUMNAS (IZQUIERDA ANCHA / DERECHA ESTRECHA) ===
    c_left, c_right = st.columns([1.8, 1], gap="medium") 
    
    # === COLUMNA IZQUIERDA (EXPEDIENTE + CONTEXTO) ===
    with c_left:
        
        # FICHA PACIENTE (DISEÑO SOLICITADO: ICONO IZQ, BADGE DER)
        st.markdown(f"""<div class="card" style="flex-direction:row; align-items:center;">
<div style="display:flex; align-items:center; gap:20px; flex-grow:1;">
<div style="background:#F0F2F5; width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2rem; color:{CEMP_DARK};">👤</div>
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

        # CONTEXTO POBLACIONAL (HTML COMPACTO SIN ESPACIOS)
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

    # === COLUMNA DERECHA (HALLAZGOS + PROBABILIDAD) ===
    with c_right:
        
        # HALLAZGOS
        st.markdown(f"""<div class="card" style="border-left:5px solid {insight_bd}; justify-content:center;">
    <span class="card-header" style="color:{insight_bd}; margin-bottom:10px;">HALLAZGOS CLAVE</span>
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h3 style="margin:0; color:{CEMP_DARK}; font-size:1.1rem; line-height:1.4;">{insight_txt}</h3>
        <div style="font-size:1.8rem;">{'⚠️' if alerts else '✅'}</div>
    </div>
</div>""", unsafe_allow_html=True)
        
        # PROBABILIDAD IA
        fig, ax = plt.subplots(figsize=(3, 3)) 
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        # Donut Chart
        ax.pie([prob, 1-prob], colors=[risk_color, '#F4F6F9'], startangle=90, counterclock=False, wedgeprops=dict(width=0.18, edgecolor='none'))
        chart_html = fig_to_html(fig)
        plt.close(fig)

        # Usamos CSS para forzar que esta tarjeta crezca y ocupe el espacio restante visualmente si es necesario
        st.markdown(f"""<div class="card" style="text-align:center; align-items:center; justify-content:center; flex-grow:1;">
<span class="card-header" style="margin-bottom:15px;">PROBABILIDAD IA</span>
<div style="position:relative; display:inline-block; margin: auto;">
{chart_html}
<div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-size:2.5rem; font-weight:800; color:{CEMP_DARK}; letter-spacing:-1px;">
{prob*100:.1f}%
</div>
</div>
<div style="font-size:0.8rem; color:#888; margin-top:15px;">Confianza: <strong>Alta</strong> <br> Umbral: {threshold}</div>
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
