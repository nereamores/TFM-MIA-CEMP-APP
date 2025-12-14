import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="CEMP Precision Care",
    page_icon="🩺",
    layout="wide"
)

# --- 2. PALETA DE COLORES CEMP ---
CEMP_PINK = "#E97F87"
CEMP_DARK = "#2C3E50"
GOOD_TEAL = "#4DB6AC"
SOFT_BG = "#F4F6F9"
RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

# --- 3. CSS AVANZADO (ESTILO ENTERPRISE) ---
st.markdown(f"""
    <style>
    /* Ocultar elementos por defecto */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .block-container {{padding-top: 2rem; padding-bottom: 3rem;}}
    
    /* LOGO SIDEBAR */
    .cemp-logo {{ 
        font-family: 'Helvetica Neue', sans-serif; font-weight: 800; 
        font-size: 2.2rem; color: {CEMP_DARK}; letter-spacing: -1px; margin-bottom: 0;
    }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}
    
    /* TARJETAS FLOTANTES (EFECTO SOMBRA SUAVE) */
    .dashboard-card {{
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03); /* Sombra muy sutil */
        border: 1px solid rgba(0,0,0,0.02);
        height: 100%;
    }}
    
    /* KPI CARDS (En Sidebar) */
    .kpi-card {{
        background: white; border-left: 4px solid {CEMP_PINK};
        padding: 12px; border-radius: 6px; margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }}
    .kpi-val {{ font-size: 1.5rem; font-weight: bold; color: {CEMP_DARK}; }}
    .kpi-lbl {{ font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px;}}

    /* BARRAS DE PROGRESO (BENCHMARKING) */
    .bar-bg {{
        width: 100%; height: 8px; background-color: #E0E5EC;
        border-radius: 4px; position: relative; margin-top: 10px; margin-bottom: 25px;
    }}
    .bar-fill {{ height: 100%; border-radius: 4px; opacity: 0.9; }}
    .bar-marker {{
        position: absolute; top: -5px; width: 4px; height: 18px;
        background-color: {CEMP_DARK}; border: 1px solid white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }}
    .bar-label {{
        position: absolute; top: -22px; transform: translateX(-50%);
        font-size: 0.8rem; font-weight: bold; color: {CEMP_DARK};
    }}
    
    /* TARJETAS DE RECOMENDACIÓN */
    .rec-card {{
        background-color: #FAFAFA; border: 1px solid #EEE;
        padding: 15px; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: start;
    }}
    .rec-icon {{ font-size: 1.2rem; margin-right: 10px; min-width: 25px; }}
    </style>
""", unsafe_allow_html=True)

# --- 4. LÓGICA MOCK (MODELO SIMULADO) ---
if 'model' not in st.session_state:
    class MockModel:
        def predict_proba(self, X):
            # Simulación: Glucosa y BMI pesan mucho
            score = (X[0] * 0.5) + (X[1] * 0.4) + (X[3] * 0.1) 
            prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

# Función para generar el texto inteligente del rectángulo derecho
def generar_smart_insight(glucose, bmi, insulin):
    alerts = []
    color = GOOD_TEAL
    icon = "✅"
    
    if glucose > 120:
        alerts.append("Hiperglucemia")
        color = "#F39C12" # Naranja alerta
        icon = "⚠️"
    if glucose > 140:
        color = CEMP_PINK # Rojo CEMP peligro
        
    if bmi > 30:
        alerts.append("Obesidad G1")
        
    homa = (glucose * insulin) / 405
    if homa > 2.5:
        alerts.append("Resistencia Insulínica")
        
    if not alerts:
        return "Paciente metabólicamente estable.", color, icon
    else:
        return " • ".join(alerts), color, icon

# --- 5. BARRA LATERAL (INPUTS) ---
with st.sidebar:
    st.markdown('<div class="cemp-logo">CEMP<span>.</span>AI</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#999; font-size:0.75rem; margin-bottom:30px;'>CLINICAL DECISION SUPPORT SYSTEM</p>", unsafe_allow_html=True)
    
    st.markdown("### 🧬 Biomarcadores")
    # Los sliders serán rosas automáticamente por el config.toml
    glucose = st.slider("Glucosa (mg/dL)", 50, 250, 120)
    bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 28.5)
    insulin = st.slider("Insulina (mu U/ml)", 0, 600, 100)
    age = st.slider("Edad (años)", 18, 90, 45)
    
    with st.expander("Factores Secundarios"):
        pregnancies = st.slider("Embarazos", 0, 15, 1)
        dpf = st.slider("Función Pedigrí", 0.0, 2.5, 0.5)

    st.markdown("---")
    
    # KPIs Rápidos
    homa_ir = glucose * insulin / 405
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{homa_ir:.1f}</div><div class="kpi-lbl">HOMA-IR</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{bmi:.1f}</div><div class="kpi-lbl">BMI</div></div>', unsafe_allow_html=True)

# --- 6. ÁREA PRINCIPAL ---
input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
prob = st.session_state.model.predict_proba(input_data)[0][1]
is_high = prob > 0.27

# Título Principal
st.markdown(f"<h1 style='color:{CEMP_DARK}; margin-bottom:5px;'>Perfil de Riesgo Metabólico</h1>", unsafe_allow_html=True)

# Pestañas
tab1, tab2, tab3 = st.tabs(["Panel General", "Factores (SHAP)", "Protocolo Clínico"])

# --- TAB 1: DASHBOARD COMPLETO ---
with tab1:
    st.write("") # Espaciador
    
    # A) RECTÁNGULOS SUPERIORES (FICHA + SMART INSIGHT)
    insight_text, insight_color, insight_icon = generar_smart_insight(glucose, bmi, insulin)
    
    col_top1, col_top2 = st.columns([1, 1], gap="medium")
    
    # 1. Ficha del Paciente
    with col_top1:
        st.markdown(f"""
            <div class="dashboard-card" style="display:flex; justify-content:space-between; align-items:center; padding: 15px 25px;">
                <div>
                    <span style="color:#999; font-size:0.7rem; font-weight:bold; letter-spacing:1px;">EXPEDIENTE MÉDICO</span>
                    <h3 style="margin:0; color:{CEMP_DARK}; font-size:1.3rem;">Paciente #8842-X</h3>
                    <div style="margin-top:5px; color:#666; font-size:0.85rem;">📅 Revisión: <b>14 Dic 2025</b></div>
                </div>
                <div style="background:#F0F2F5; width:45px; height:45px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.2rem;">👤</div>
            </div>
        """, unsafe_allow_html=True)
        
    # 2. Smart Insight (Dinámico)
    with col_top2:
        st.markdown(f"""
            <div class="dashboard-card" style="display:flex; justify-content:space-between; align-items:center; padding: 15px 25px; border-left: 6px solid {insight_color};">
                <div>
                    <span style="color:{insight_color}; font-size:0.7rem; font-weight:bold; letter-spacing:1px;">HALLAZGOS CLAVE</span>
                    <h3 style="margin:0; color:{CEMP_DARK}; font-size:1.1rem; line-height:1.3;">{insight_text}</h3>
                </div>
                <div style="font-size:1.8rem;">{insight_icon}</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("") # Espacio vertical
    
    # B) GRÁFICOS PRINCIPALES
    c_left, c_right = st.columns([1, 2], gap="medium")
    
    # Columna Izquierda: Probabilidad (Donut Chart)
    with c_left:
        st.markdown('<div class="dashboard-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#555; margin-bottom:10px;">Probabilidad IA</h4>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        ring_color = CEMP_PINK if is_high else GOOD_TEAL
        ax.pie([prob, 1-prob], colors=[ring_color, '#F0F0F0'], startangle=90, 
               counterclock=False, wedgeprops=dict(width=0.12, edgecolor='white'))
        
        ax.text(0, 0, f"{prob*100:.1f}%", ha='center', va='center', fontsize=28, fontweight='bold', color=CEMP_DARK)
        st.pyplot(fig, use_container_width=True)
        
        if is_high:
            st.markdown(f"<div style='color:{CEMP_PINK}; font-weight:bold; margin-top:-10px;'>RIESGO ELEVADO</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:{GOOD_TEAL}; font-weight:bold; margin-top:-10px;'>BAJO RIESGO</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Columna Derecha: Benchmarking Poblacional (Barras con Gradiente)
    with c_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#555; margin-bottom:20px;">Contexto Poblacional</h4>', unsafe_allow_html=True)
        
        # Barra Glucosa
        g_pos = min(100, max(0, (glucose - 60) / (200 - 60) * 100))
        st.markdown(f"""
            <div style="margin-bottom:5px; font-size:0.85rem; color:#666; font-weight:bold;">GLUCOSA BASAL <span style="font-weight:normal">({glucose} mg/dL)</span></div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: 100%; background: {RISK_GRADIENT};"></div>
                <div class="bar-marker" style="left: {g_pos}%;"></div>
                <div class="bar-label" style="left: {g_pos}%;">{glucose}</div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#999; margin-top:-20px; margin-bottom:25px;">
                <span>Hipoglucemia</span><span>Normal</span><span>Prediabetes</span><span>Diabetes</span>
            </div>
        """, unsafe_allow_html=True)

        # Barra BMI
        b_pos = min(100, max(0, (bmi - 18) / (40 - 18) * 100))
        st.markdown(f"""
            <div style="margin-bottom:5px; font-size:0.85rem; color:#666; font-weight:bold;">ÍNDICE DE MASA CORPORAL <span style="font-weight:normal">({bmi})</span></div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: 100%; background: {RISK_GRADIENT};"></div>
                <div class="bar-marker" style="left: {b_pos}%;"></div>
                <div class="bar-label" style="left: {b_pos}%;">{bmi}</div>
            </div>
             <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#999; margin-top:-20px;">
                <span>Peso Sano</span><span>Sobrepeso</span><span>Obesidad G1</span><span>Obesidad G2</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: EXPLICABILIDAD (SHAP ELEGANTE) ---
with tab2:
    st.write("")
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Drivers de la Predicción")
    st.caption("Análisis de contribución de variables (SHAP values)")
    
    features = ["Glucosa", "BMI", "Edad", "Insulina", "Genética"]
    vals = [(glucose-100)/100, (bmi-25)/50, -0.1, 0.05, 0.02]
    
    fig_s, ax_s = plt.subplots(figsize=(10, 4))
    fig_s.patch.set_facecolor('none')
    ax_s.set_facecolor('none')
    
    y = np.arange(len(features))
    # Colores: Rosa CEMP para Riesgo, Gris para Protección
    colors = [CEMP_PINK if x > 0 else "#BDC3C7" for x in vals]
    
    ax_s.barh(y, vals, color=colors, height=0.6, edgecolor='none')
    ax_s.set_yticks(y)
    ax_s.set_yticklabels(features, fontsize=11, color=CEMP_DARK, fontweight='bold')
    ax_s.axvline(0, color='#eee')
    
    # Limpiar bordes gráfico
    for spine in ax_s.spines.values():
        spine.set_visible(False)
    ax_s.tick_params(axis='x', colors='#999')
    ax_s.tick_params(axis='y', length=0)
    
    st.pyplot(fig_s)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: PROTOCOLO DE ACCIÓN ---
with tab3:
    st.write("")
    c_rec1, c_rec2 = st.columns(2, gap="medium")
    
    with c_rec1:
        st.markdown("#### 🥗 Plan Nutricional")
        if glucose > 120:
             st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-icon" style="color:#F39C12">⚠️</div>
                    <div>
                        <b>Control Glucémico Estricto</b><br>
                        <span style="color:#666; font-size:0.9rem;">Reducir carga glucémica. Evitar picos de insulina post-prandiales.</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="rec-card">
                <div class="rec-icon" style="color:{GOOD_TEAL}">💧</div>
                <div>
                    <b>Hidratación y Electrolitos</b><br>
                    <span style="color:#666; font-size:0.9rem;">Aumentar ingesta hídrica a 2.5L/día para soporte metabólico.</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c_rec2:
        st.markdown("#### 🩺 Seguimiento Clínico")
        if is_high:
            st.markdown(f"""
                <div class="rec-card" style="border-left: 4px solid {CEMP_PINK}; background-color:#FFF5F6;">
                    <div class="rec-icon" style="color:{CEMP_PINK}">🔴</div>
                    <div>
                        <b>Protocolo de Alto Riesgo</b><br>
                        <span style="color:#666; font-size:0.9rem;">Solicitar HbA1c urgente. Monitorización continua (CGM) sugerida por 14 días.</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="rec-card" style="border-left: 4px solid {GOOD_TEAL};">
                    <div class="rec-icon" style="color:{GOOD_TEAL}">🟢</div>
                    <div>
                        <b>Control Rutinario</b><br>
                        <span style="color:#666; font-size:0.9rem;">Mantener hábitos actuales. Repetir analítica en 12 meses.</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
