import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="CEMP Precision Care", page_icon="🩺", layout="wide")

# --- PALETA DE COLORES CEMP (ARMONIZADA) ---
CEMP_PINK = "#E97F87"      # Tu color de marca
CEMP_DARK = "#2C3E50"      # Gris oscuro profesional
CEMP_SOFT_GREY = "#E0E5EC" # Para fondos de barras
GOOD_TEAL = "#4DB6AC"      # Un verde azulado elegante que combina con el rosa
RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

st.markdown(f"""
    <style>
    /* LIMPIEZA INTERFAZ */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .block-container {{padding-top: 1rem; padding-bottom: 2rem;}}
    
    /* LOGO Y TÍTULOS */
    .cemp-logo {{ 
        font-family: 'Helvetica Neue', sans-serif; font-weight: 800; 
        font-size: 2.2rem; color: {CEMP_DARK}; letter-spacing: -1px;
    }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}
    
    /* TARJETAS FLOTANTES (EFECTO ENTERPRISE) */
    .dashboard-card {{
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.03);
        margin-bottom: 15px;
    }}
    
    /* KPI CARDS (En Sidebar) */
    .kpi-card {{
        background: white; border-left: 4px solid {CEMP_PINK};
        padding: 10px; border-radius: 4px; margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .kpi-val {{ font-size: 1.4rem; font-weight: bold; color: {CEMP_DARK}; }}
    .kpi-lbl {{ font-size: 0.75rem; color: #888; text-transform: uppercase; }}

    /* ESTILO DE BARRAS PERSONALIZADAS */
    .bar-bg {{
        width: 100%; height: 10px; background-color: {CEMP_SOFT_GREY};
        border-radius: 5px; position: relative; margin-top: 8px; margin-bottom: 20px;
    }}
    .bar-fill {{
        height: 100%; border-radius: 5px; opacity: 0.8;
    }}
    .bar-marker {{
        position: absolute; top: -6px; width: 4px; height: 22px;
        background-color: {CEMP_DARK}; border: 1px solid white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }}
    .bar-label {{
        position: absolute; top: -25px; transform: translateX(-50%);
        font-size: 0.85rem; font-weight: bold; color: {CEMP_DARK};
        background: white; padding: 2px 6px; border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    
    /* ALERTAS SUAVES */
    .rec-card {{
        background-color: #FAFAFA; border: 1px solid #EEE;
        padding: 15px; border-radius: 8px; margin-bottom: 10px;
    }}
    .rec-icon {{ color: {CEMP_PINK}; font-size: 1.2rem; margin-right: 8px; }}
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

# --- SIDEBAR LIMPIO ---
with st.sidebar:
    st.markdown('<div class="cemp-logo">CEMP<span>.</span>AI</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#999; font-size:0.8rem;'>CLINICAL DECISION SUPPORT</p>", unsafe_allow_html=True)
    
    # Sliders Rosas (Automático por config.toml)
    st.markdown("### 🧬 Biomarcadores")
    glucose = st.slider("Glucosa (mg/dL)", 50, 250, 120)
    bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 28.5)
    insulin = st.slider("Insulina (mu U/ml)", 0, 600, 100)
    age = st.slider("Edad", 18, 90, 45)
    
    with st.expander("Opciones Avanzadas"):
        pregnancies = st.slider("Embarazos", 0, 15, 1)
        dpf = st.slider("Función Pedigrí", 0.0, 2.5, 0.5)

    st.markdown("---")
    
    # KPIs en gris y rosa (sin cajas de colores chillones)
    homa_ir = glucose * insulin / 405
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{homa_ir:.1f}</div><div class="kpi-lbl">HOMA-IR</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{bmi:.1f}</div><div class="kpi-lbl">BMI</div></div>', unsafe_allow_html=True)

# --- ÁREA PRINCIPAL ---
input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
prob = st.session_state.model.predict_proba(input_data)[0][1]
is_high = prob > 0.27

# Encabezado minimalista
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.title("Perfil de Riesgo Metabólico")
with col_h2:
    # Badge de estado elegante (No semáforo)
    if is_high:
        st.markdown(f'<div style="margin-top:20px; text-align:right; color:{CEMP_PINK}; font-weight:bold; font-size:1.2rem;">🔴 RIESGO ELEVADO</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="margin-top:20px; text-align:right; color:{GOOD_TEAL}; font-weight:bold; font-size:1.2rem;">🟢 BAJO RIESGO</div>', unsafe_allow_html=True)

# PESTAÑAS
tab1, tab2, tab3 = st.tabs(["Panel General", "Factores (SHAP)", "Protocolo"])

# --- TAB 1: VISUALIZACIÓN UNIFICADA ---
with tab1:
    c_left, c_right = st.columns([1, 2], gap="medium")
    
    # 1. IZQUIERDA: GAUGE (DONUT)
    with c_left:
        st.markdown('<div class="dashboard-card" style="text-align:center; height:100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#555; margin-bottom:20px;">Probabilidad IA</h4>', unsafe_allow_html=True)
        
        # Gráfico Donut Matplotlib limpio
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        # Color del anillo: Rosa si es alto, Teal si es bajo (mantiene armonía)
        ring_color = CEMP_PINK if is_high else GOOD_TEAL
        
        ax.pie([prob, 1-prob], colors=[ring_color, '#F0F0F0'], startangle=90, 
               counterclock=False, wedgeprops=dict(width=0.1, edgecolor='white'))
        
        # Texto central
        ax.text(0, 0, f"{prob*100:.1f}%", ha='center', va='center', fontsize=32, fontweight='bold', color=CEMP_DARK)
        
        st.pyplot(fig, use_container_width=True)
        st.caption("Score basado en 6 variables clínicas.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. DERECHA: BENCHMARKING (BARRAS ESTILO CEMP)
    with c_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#555;">Contexto Poblacional</h4>', unsafe_allow_html=True)
        st.write("")
        
        # --- BARRA GLUCOSA ---
        st.markdown(f"<small style='color:#888'>NIVEL DE GLUCOSA ({glucose} mg/dL)</small>", unsafe_allow_html=True)
        # Calculamos posición (0-100%)
        g_pos = min(100, max(0, (glucose - 60) / (200 - 60) * 100))
        
        # HTML DE LA BARRA (Usando el gradiente definido en CSS variables)
        st.markdown(f"""
            <div class="bar-bg">
                <div class="bar-fill" style="width: 100%; background: {RISK_GRADIENT};"></div>
                <div class="bar-marker" style="left: {g_pos}%;"></div>
                <div class="bar-label" style="left: {g_pos}%;">{glucose}</div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#AAA; margin-top:-15px; margin-bottom:20px;">
                <span>Normal</span><span>Prediabetes</span><span>Diabetes</span>
            </div>
        """, unsafe_allow_html=True)

        # --- BARRA BMI ---
        st.markdown(f"<small style='color:#888'>ÍNDICE DE MASA CORPORAL ({bmi})</small>", unsafe_allow_html=True)
        b_pos = min(100, max(0, (bmi - 18) / (40 - 18) * 100))
        
        st.markdown(f"""
            <div class="bar-bg">
                <div class="bar-fill" style="width: 100%; background: {RISK_GRADIENT};"></div>
                <div class="bar-marker" style="left: {b_pos}%;"></div>
                <div class="bar-label" style="left: {b_pos}%;">{bmi}</div>
            </div>
             <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#AAA; margin-top:-15px;">
                <span>Peso Sano</span><span>Sobrepeso</span><span>Obesidad</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: EXPLICABILIDAD (SHAP MONOCROMO) ---
with tab2:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Drivers de la Predicción")
    
    features = ["Glucosa", "BMI", "Edad", "Insulina", "Genética"]
    vals = [(glucose-100)/100, (bmi-25)/50, -0.1, 0.05, 0.02]
    
    fig_s, ax_s = plt.subplots(figsize=(10, 3))
    fig_s.patch.set_facecolor('none')
    ax_s.set_facecolor('none')
    
    y = np.arange(len(features))
    # AQUÍ ESTÁ LA CLAVE: Rosa para riesgo, Gris para protección (NO azul/rojo estándar)
    colors = [CEMP_PINK if x > 0 else "#BDC3C7" for x in vals]
    
    ax_s.barh(y, vals, color=colors, height=0.5, edgecolor='none')
    ax_s.set_yticks(y)
    ax_s.set_yticklabels(features, fontsize=10, color=CEMP_DARK, fontweight='bold')
    ax_s.axvline(0, color='#eee')
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)
    ax_s.spines['bottom'].set_visible(False)
    ax_s.spines['left'].set_visible(False)
    ax_s.tick_params(axis='x', colors='#999')
    
    st.pyplot(fig_s)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: RECOMENDACIONES (ESTILO TARJETAS) ---
with tab3:
    c_rec1, c_rec2 = st.columns(2)
    
    with c_rec1:
        st.markdown("#### 🥗 Nutrición")
        if glucose > 120:
             st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-icon">⚠️</span> <b>Control Glucémico:</b> <br>
                    <span style="color:#666; font-size:0.9rem;">Reducir carga glucémica. Priorizar fibra.</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="rec-card">
                <span class="rec-icon">💧</span> <b>Hidratación:</b> <br>
                <span style="color:#666; font-size:0.9rem;">Aumentar ingesta hídrica a 2.5L/día.</span>
            </div>
        """, unsafe_allow_html=True)

    with c_rec2:
        st.markdown("#### 🩺 Seguimiento")
        if is_high:
            st.markdown(f"""
                <div class="rec-card" style="border-left: 4px solid {CEMP_PINK};">
                    <span class="rec-icon">🔴</span> <b>Confirmación Diagnóstica:</b> <br>
                    <span style="color:#666; font-size:0.9rem;">Solicitar HbA1c en < 15 días.</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="rec-card" style="border-left: 4px solid {GOOD_TEAL};">
                    <span class="rec-icon">🟢</span> <b>Control Rutinario:</b> <br>
                    <span style="color:#666; font-size:0.9rem;">Repetir analítica en 12 meses.</span>
                </div>
            """, unsafe_allow_html=True)
