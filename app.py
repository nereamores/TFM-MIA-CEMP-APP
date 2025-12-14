import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="CEMP AI", page_icon="🩺", layout="wide")

# --- COLORES ---
CEMP_PINK = "#E97F87"
CEMP_DARK = "#2C3E50"
GOOD_TEAL = "#4DB6AC"
# Gradiente exacto de la imagen que te gustaba
RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

# --- CSS ENTERPRISE (ESTILOS) ---
st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .block-container {{padding-top: 2rem; padding-bottom: 3rem;}}
    
    /* LOGO */
    .cemp-logo {{ font-family: 'Helvetica', sans-serif; font-weight: 800; font-size: 2.2rem; color: {CEMP_DARK}; margin:0; }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}
    
    /* TARJETAS (ESTILO UNIFICADO) */
    .card {{
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04); /* Sombra suave */
        border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 20px;
        height: 100%;
    }}
    
    /* KPI SIDEBAR */
    .kpi-box {{
        background: white; border-left: 4px solid {CEMP_PINK};
        padding: 12px; border-radius: 6px; margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* BARRAS DE PROGRESO CEMP (CSS PURO) */
    .bar-container {{
        position: relative;
        width: 100%;
        margin-top: 10px;
        margin-bottom: 30px;
    }}
    .bar-bg {{ 
        background: #F0F2F5; 
        height: 10px; 
        border-radius: 5px; 
        width: 100%;
        overflow: hidden; /* Mantiene el gradiente dentro */
    }}
    .bar-fill {{ 
        height: 100%; 
        width: 100%; 
        border-radius: 5px; 
        opacity: 0.9; 
    }}
    .bar-marker {{ 
        position: absolute; 
        top: -6px; 
        width: 4px; 
        height: 22px; 
        background: {CEMP_DARK}; 
        border: 2px solid white; 
        border-radius: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        z-index: 10;
        transition: left 0.3s ease; /* Movimiento suave */
    }}
    .bar-txt {{ 
        position: absolute; 
        top: -25px; 
        transform: translateX(-50%); 
        font-size: 0.85rem; 
        font-weight: bold; 
        color: {CEMP_DARK}; 
        background: white;
        padding: 2px 6px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    
    </style>
""", unsafe_allow_html=True)

# --- HELPER: CONVERTIR GRÁFICOS MATPLOTLIB A HTML ---
def fig_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%; object-fit:contain;">'

# --- MODELO MOCK ---
if 'model' not in st.session_state:
    class MockModel:
        def predict_proba(self, X):
            score = (X[0]*0.5) + (X[1]*0.4) + (X[3]*0.1) 
            prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

# --- SIDEBAR ---
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
    homa = glucose * insulin / 405
    c1, c2 = st.columns(2)
    # HTML en una sola línea para evitar errores de identación
    with c1: st.markdown(f'<div class="kpi-box"><div style="font-size:1.4rem; font-weight:bold; color:{CEMP_DARK}">{homa:.1f}</div><div style="font-size:0.7rem; color:#888">HOMA-IR</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-box"><div style="font-size:1.4rem; font-weight:bold; color:{CEMP_DARK}">{bmi:.1f}</div><div style="font-size:0.7rem; color:#888">BMI</div></div>', unsafe_allow_html=True)

# --- MAIN ---
input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
prob = st.session_state.model.predict_proba(input_data)[0][1]
is_high = prob > 0.27
risk_color = CEMP_PINK if is_high else GOOD_TEAL
risk_label = "ALTO RIESGO" if is_high else "BAJO RIESGO"
risk_icon = "🔴" if is_high else "🟢"

# CABECERA
c_tit, c_bad = st.columns([3,1])
with c_tit: st.markdown(f"<h1 style='color:{CEMP_DARK}; margin:0;'>Perfil de Riesgo Metabólico</h1>", unsafe_allow_html=True)
with c_bad: st.markdown(f"<div style='text-align:right; margin-top:10px; color:{risk_color}; font-weight:bold; font-size:1.2rem;'>{risk_icon} {risk_label}</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Panel General", "Factores (SHAP)", "Protocolo"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.write("")
    
    # 1. HALLAZGOS Y FICHA
    alerts = []
    if glucose > 120: alerts.append("Hiperglucemia")
    if bmi > 30: alerts.append("Obesidad")
    if homa > 2.5: alerts.append("Resistencia Insulina")
    insight_txt = " • ".join(alerts) if alerts else "Paciente estable"
    insight_bd = CEMP_PINK if alerts else GOOD_TEAL

    col_top1, col_top2 = st.columns(2, gap="medium")
    
    with col_top1:
        # String HTML sin espacios al inicio de cada línea
        st.markdown(f"""<div class="card" style="display:flex; justify-content:space-between; align-items:center;">
    <div>
        <span style="color:#999; font-size:0.7rem; font-weight:bold;">EXPEDIENTE</span>
        <h3 style="margin:0; color:{CEMP_DARK};">Paciente #8842-X</h3>
        <div style="font-size:0.8rem; color:#666;">📅 14 Dic 2025</div>
    </div>
    <div style="background:#F0F2F5; padding:10px; border-radius:50%; font-size:1.5rem;">👤</div>
</div>""", unsafe_allow_html=True)
        
    with col_top2:
        st.markdown(f"""<div class="card" style="display:flex; justify-content:space-between; align-items:center; border-left:5px solid {insight_bd};">
    <div>
        <span style="color:{insight_bd}; font-size:0.7rem; font-weight:bold;">HALLAZGOS CLAVE</span>
        <h3 style="margin:0; color:{CEMP_DARK}; font-size:1.1rem;">{insight_txt}</h3>
    </div>
    <div style="font-size:1.5rem;">{'⚠️' if alerts else '✅'}</div>
</div>""", unsafe_allow_html=True)

    # 2. GRÁFICOS INCRUSTADOS
    c_left, c_right = st.columns([1, 2], gap="medium")
    
    # IZQUIERDA: DONUT
    with c_left:
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.pie([prob, 1-prob], colors=[risk_color, '#F0F0F0'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15, edgecolor='white'))
        ax.text(0, 0, f"{prob*100:.1f}%", ha='center', va='center', fontsize=26, fontweight='bold', color=CEMP_DARK)
        chart_html = fig_to_html(fig)
        plt.close(fig)

        st.markdown(f"""<div class="card" style="text-align:center;">
    <h4 style="color:#555; margin-bottom:0;">Probabilidad IA</h4>
    {chart_html}
    <div style="font-size:0.8rem; color:#888; margin-top:-10px;">Certeza del modelo</div>
</div>""", unsafe_allow_html=True)

    # DERECHA: CONTEXTO (BARRAS CSS CORREGIDAS)
    with c_right:
        # Cálculos de posición (0 a 100%)
        g_pos = min(100, max(0, (glucose - 60) / 1.4))
        b_pos = min(100, max(0, (bmi - 18) / 0.22))
        
        # AQUÍ ESTÁ EL ARREGLO: 
        # El string HTML está pegado a la izquierda, sin sangría.
        # Streamlit lo renderizará como gráficos visuales dentro de la caja blanca.
        st.markdown(f"""<div class="card">
<h4 style="color:#555; margin-bottom:25px;">Contexto Poblacional</h4>

<div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">GLUCOSA <span style="font-weight:normal">({glucose} mg/dL)</span></div>
<div class="bar-container">
    <div class="bar-bg">
        <div class="bar-fill" style="background:{RISK_GRADIENT};"></div>
    </div>
    <div class="bar-marker" style="left: {g_pos}%;"></div>
    <div class="bar-txt" style="left: {g_pos}%;">{glucose}</div>
</div>

<div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px; margin-top:20px;">BMI <span style="font-weight:normal">({bmi})</span></div>
<div class="bar-container">
    <div class="bar-bg">
        <div class="bar-fill" style="background:{RISK_GRADIENT};"></div>
    </div>
    <div class="bar-marker" style="left: {b_pos}%;"></div>
    <div class="bar-txt" style="left: {b_pos}%;">{bmi}</div>
</div>

<div style="display:flex; justify-content:space-between; font-size:0.6rem; color:#AAA; margin-top:10px; border-top:1px solid #EEE; padding-top:10px;">
    <span>Sano</span>
    <span>Riesgo</span>
    <span>Peligro</span>
</div>
</div>""", unsafe_allow_html=True)

# --- TAB 2: SHAP ---
with tab2:
    st.write("")
    features = ["Glucosa", "BMI", "Edad", "Insulina"]
    vals = [(glucose-100)/100, (bmi-25)/50, -0.1, 0.05]
    colors = [CEMP_PINK if x>0 else "#BDC3C7" for x in vals]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.barh(features, vals, color=colors, height=0.5)
    ax.axvline(0, color='#eee')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    chart_html = fig_to_html(fig)
    plt.close(fig)

    st.markdown(f"""<div class="card">
<h4 style="color:#555;">Drivers de la Predicción</h4>
{chart_html}
</div>""", unsafe_allow_html=True)

# --- TAB 3: PROTOCOLO ---
with tab3:
    st.write("")
    st.info("💡 Módulo de recomendaciones clínicas.")
