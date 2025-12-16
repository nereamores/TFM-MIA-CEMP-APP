import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# --- 1. CONFIGURACIÓN ---
st.set_page_config(
    page_title="DIABETES.NME", 
    page_icon="🩺", 
    layout="wide"
)

# --- 2. COLORES ---
CEMP_PINK = "#E97F87"
CEMP_DARK = "#2C3E50" # Azul muy oscuro
GOOD_TEAL = "#4DB6AC"
SLIDER_GRAY = "#BDC3C7"
OPTIMAL_GREEN = "#8BC34A" # Verde lima (Referencia F2)
NOTE_GRAY_BG = "#F8F9FA"  # Fondo gris para notas
NOTE_GRAY_TEXT = "#6C757D" # Texto gris para notas

# --- GRADIENTES SUAVES PERO CLÍNICOS ---
# Ajustados para que la transición de color ocurra cerca de los puntos de corte

# BMI (Escala 10-50)
BMI_GRADIENT = "linear-gradient(90deg, #81D4FA 0%, #4DB6AC 25%, #FFF176 40%, #FFB74D 55%, #E97F87 70%, #880E4F 100%)"

# GLUCOSA (Escala 50-350)
# Cortes en 140 (30%) y 200 (50%)
GLUCOSE_GRADIENT = "linear-gradient(90deg, #4DB6AC 0%, #4DB6AC 25%, #FFF176 35%, #FFB74D 45%, #E97F87 55%, #880E4F 100%)"

# Genérico
RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

# --- 3. CSS (ESTILOS AVANZADOS) ---
# IMPORTANTE: Usamos {{ y }} para escapar las llaves en el f-string
st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .block-container {{
        max-width: 1250px; 
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }}
    
    /* LOGO PERSONALIZADO */
    .cemp-logo {{ 
        font-family: 'Helvetica', sans-serif; 
        font-weight: 800; 
        font-size: 1.8rem; 
        color: {CEMP_DARK}; 
        margin: 0; 
    }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}

    /* === ESTILO SLIDER GENERAL === */
    .stSlider {{
        padding-top: 0px !important;
        padding-bottom: 10px !important;
    }}

    /* === ESTILO DEL DESPLEGABLE (EXPANDER) === */
    div[data-testid="stExpander"] details > summary {{
        background-color: rgba(233, 127, 135, 0.1) !important;
        border: 1px solid rgba(233, 127, 135, 0.2) !important;
        border-radius: 8px !important;
        color: {CEMP_DARK} !important;
        font-weight: 700 !important;
        transition: background-color 0.3s;
    }}
    
    div[data-testid="stExpander"] details > summary:hover {{
        background-color: rgba(233, 127, 135, 0.2) !important;
        color: {CEMP_DARK} !important;
    }}

    div[data-testid="stExpander"] details > summary svg {{
        fill: {CEMP_DARK} !important;
        color: {CEMP_DARK} !important;
    }}
    
    div[data-testid="stExpander"] details[open] > div {{
        border-left: 1px solid rgba(233, 127, 135, 0.2);
        border-right: 1px solid rgba(233, 127, 135, 0.2);
        border-bottom: 1px solid rgba(233, 127, 135, 0.2);
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
    }}

    /* === INPUTS BARRA LATERAL === */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
        padding: 0px 5px;
        font-size: 0.9rem;
        text-align: center;
        color: {CEMP_DARK};
        font-weight: 800;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #ddd;
    }}
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {{
        vertical-align: middle;
    }}

    /* === CAJA DE CÁLCULOS (SIDEBAR) === */
    .calc-box {{
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 12px 15px;
        border: 1px solid #EEE;
        margin-top: 5px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }}
    .calc-label {{
        font-size: 0.75rem; 
        color: #888; 
        font-weight: 600; 
        text-transform: uppercase;
    }}
    .calc-value {{
        font-size: 1rem; 
        color: {CEMP_DARK}; 
        font-weight: 800;
    }}
    
    /* === TARJETAS === */
    .card {{
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 15px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 300px; 
    }}
    
    .card-auto {{
        min-height: auto !important;
        height: 100%;
    }}
    
    .card-header {{
        color: #999;
        font-size: 0.75rem;
        font-weight: bold;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }}

    /* GRÁFICOS DE BARRAS */
    .bar-container {{
        position: relative; width: 100%; margin-top: 20px; margin-bottom: 30px;
    }}
    .bar-bg {{ background: #F0F2F5; height: 12px; border-radius: 6px; width: 100%; overflow: hidden; }}
    
    .bar-fill {{ height: 100%; width: 100%; background: {RISK_GRADIENT}; border-radius: 6px; opacity: 1; }}
    .bar-fill-bmi {{ height: 100%; width: 100%; background: {BMI_GRADIENT}; border-radius: 6px; opacity: 1; }}
    .bar-fill-glucose {{ height: 100%; width: 100%; background: {GLUCOSE_GRADIENT}; border-radius: 6px; opacity: 1; }}

    .bar-marker {{ 
        position: absolute; top: -6px; width: 4px; height: 24px; 
        background: {CEMP_DARK}; border: 1px solid white; border-radius: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10; transition: left 0.3s ease;
    }}
    .bar-txt {{ 
        position: absolute; top: -30px; transform: translateX(-50%); 
        font-size: 0.85rem; font-weight: bold; color: {CEMP_DARK}; 
        background: white; padding: 2px 8px; border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    /* LEYENDA POSICIONADA ABSOLUTAMENTE PARA ALINEACIÓN PERFECTA */
    /* He corregido el error de sintaxis aquí poniendo dobles llaves */
    .legend-container {{
        position: relative;
        width: 100%;
        height: 20px;
        margin-top: 8px;
    }}
    .legend-label {{
        position: absolute;
        transform: translateX(-50%);
        font-size: 0.7rem; 
        color: #888; 
        font-weight: 600;
        text-align: center;
        white-space: nowrap;
    }}
    
    </style>
""", unsafe_allow_html=True)

# --- 4. HELPERS ---
def fig_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%; object-fit:contain;">'

def get_help_icon(description):
    return f"""<span style="display:inline-block; width:16px; height:16px; line-height:16px; text-align:center; border-radius:50%; background:#E0E0E0; color:#777; font-size:0.7rem; font-weight:bold; cursor:help; margin-left:6px; position:relative; top:-1px;" title="{description}">?</span>"""

# --- 5. MODELO MOCK ---
if 'model' not in st.session_state:
    class MockModel:
        def predict_proba(self, X):
            score = (X[0]*0.5) + (X[1]*0.4) + (X[3]*0.1) 
            prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

# --- 6. INPUTS SINCRONIZADOS ---
def input_biomarker(label_text, min_val, max_val, default_val, key, help_text=""):
    label_html = f"**{label_text}**"
    if help_text:
        label_html += get_help_icon(help_text)
    st.markdown(label_html, unsafe_allow_html=True)
    
    c1, c2 = st.columns([2.5, 1], gap="small")
    
    input_type = type(default_val)
    min_val = input_type(min_val)
    max_val = input_type(max_val)
    step = 0.1 if input_type == float else 1

    if key not in st.session_state:
        st.session_state[key] = default_val

    def update_from_slider():
        st.session_state[key] = st.session_state[f"{key}_slider"]
        st.session_state[f"{key}_input"] = st.session_state[f"{key}_slider"] 
    
    def update_from_input():
        val = st.session_state[f"{key}_input"]
        if val < min_val: val = min_val
        if val > max_val: val = max_val
        st.session_state[key] = val
        st.session_state[f"{key}_slider"] = val 

    with c1:
        st.slider(
            label="", min_value=min_val, max_value=max_val, step=step,
            key=f"{key}_slider", value=st.session_state[key], on_change=update_from_slider, label_visibility="collapsed"
        )
    with c2:
        st.number_input(
            label="", min_value=min_val, max_value=max_val, step=step,
            key=f"{key}_input", value=st.session_state[key], on_change=update_from_input, label_visibility="collapsed"
        )
    return st.session_state[key]

# --- 7. BARRA LATERAL ---
with st.sidebar:
    st.markdown(f'<div class="cemp-logo">D<span>IA</span>BETES<span style="color:{SLIDER_GRAY}">.</span><span>NME</span></div>', unsafe_allow_html=True)
    st.caption("CLINICAL DECISION SUPPORT SYSTEM")
    st.write("")
    
    # 1. METABÓLICOS
    glucose = input_biomarker("Glucosa 2h (mg/dL)", 50, 350, 120, "gluc", "Concentración plasmática a las 2h de test de tolerancia oral.")
    insulin = input_biomarker("Insulina (µU/ml)", 0, 900, 100, "ins", "Insulina a las 2h de ingesta.")
    
    proxy_index = glucose * insulin
    st.markdown(f"""
    <div class="calc-box" style="border-left: 4px solid {CEMP_PINK};">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="calc-label">Índice RI (Glucosa x Insulina)</span>
            <span class="calc-value">{proxy_index:,.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---") 

    # 2. ANTROPOMÉTRICOS
    weight = input_biomarker("Peso (kg)", 30.0, 250.0, 70.0, "weight", "Peso corporal actual.")
    height = input_biomarker("Altura (m)", 1.00, 2.20, 1.70, "height", "Altura en metros.")
    
    bmi = weight / (height * height)
    bmi_sq = bmi ** 2
    
    st.markdown(f"""
    <div class="calc-box" style="border-left: 4px solid {CEMP_PINK};">
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <span class="calc-label">BMI (kg/m²)</span>
            <span class="calc-value">{bmi:.2f}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span class="calc-label">BMI² (Non-Linear)</span>
            <span class="calc-value">{bmi_sq:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---") 

    # 3. PACIENTE
    c_age, c_preg = st.columns(2)
    age = input_biomarker("Edad (años)", 18, 90, 45, "age")
    pregnancies = input_biomarker("Embarazos", 0, 20, 1, "preg", "Nº veces embarazada (a término o no).") 
    
    st.markdown("---") 

    # 4. DPF
    dpf = input_biomarker("Antecedentes Familiares (DPF)", 0.0, 2.5, 0.5, "dpf", "Estimación de predisposición genética por historial familiar.")

    if dpf <= 0.15:
        dpf_label, bar_color = "Carga familiar MUY BAJA", GOOD_TEAL
    elif dpf <= 0.40:
        dpf_label, bar_color = "Carga familiar BAJA", "#D4E157"
    elif dpf <= 0.80:
        dpf_label, bar_color = "Carga familiar MODERADA", "#FFB74D"
    elif dpf <= 1.20:
        dpf_label, bar_color = "Carga familiar ELEVADA", CEMP_PINK
    else:
        dpf_label, bar_color = "Carga familiar MUY ELEVADA", "#880E4F"

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:-10px; margin-bottom:2px;">
        <span style="font-size:0.8rem; font-weight:bold; color:{bar_color};">{dpf_label}</span>
        <span style="font-size:0.8rem; color:#666;">{dpf:.2f}</span>
    </div>
    <div style="width:100%; background-color:#F0F2F5; border-radius:4px; height:8px; margin-bottom:10px;">
        <div style="width:{min(100, (dpf/2.5)*100)}%; background-color:{bar_color}; height:8px; border-radius:4px; transition: width 0.3s ease, background-color 0.3s ease;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Valores basados en el estudio Pima Indians Diabetes.")


# --- 8. MAIN ---
st.markdown(f"<h1 style='color:{CEMP_DARK}; margin-bottom: 10px; font-size: 2.2rem;'>Evaluación de Riesgo Diabético</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Panel General", "Factores (SHAP)", "Protocolo"])

with tab1:
    st.write("")
    
    # --- UMBRAL ---
    with st.expander("Ajuste de Sensibilidad Clínica"):
        c_calib_1, c_calib_2 = st.columns([1, 2], gap="large")
        
        with c_calib_1:
            st.caption("Selecciona manualmente el umbral de decisión.")
            threshold = st.slider("Umbral", 0.0, 1.0, 0.27, 0.01, label_visibility="collapsed")
            
            st.markdown(f"""
            <div style="background-color:{NOTE_GRAY_BG}; margin-right: 15px; padding:15px; border-radius:8px; border:1px solid #E9ECEF; color:{NOTE_GRAY_TEXT}; font-size:0.85rem; display:flex; align-items:start; gap:10px;">
                <span style="font-size:1.1rem;">💡</span> 
                <div>
                    <strong>Criterio Técnico:</strong> Se ha seleccionado <strong>0.27</strong> como umbral óptimo (F2-Score) para priorizar la detección de casos positivos (minimizar falsos negativos).
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_calib_2:
            # --- SIMULACIÓN MATEMÁTICA ---
            x = np.linspace(-0.15, 1.25, 500)
            y_sanos = 1.9 * np.exp(-((x - 0.1)**2) / (2 * 0.11**2)) + \
                      0.5 * np.exp(-((x - 0.55)**2) / (2 * 0.15**2))
            y_enfermos = 0.35 * np.exp(-((x - 0.28)**2) / (2 * 0.1**2)) + \
                         1.4 * np.exp(-((x - 0.68)**2) / (2 * 0.16**2))
            
            fig_calib, ax_calib = plt.subplots(figsize=(6, 2.5))
            fig_calib.patch.set_facecolor('none')
            ax_calib.set_facecolor('none')
            ax_calib.fill_between(x, y_sanos, color="#BDC3C7", alpha=0.3, label="Clase 0: No Diabetes")
            ax_calib.plot(x, y_sanos, color="gray", lw=0.8, alpha=0.6)
            ax_calib.fill_between(x, y_enfermos, color=CEMP_PINK, alpha=0.3, label="Clase 1: Diabetes")
            ax_calib.plot(x, y_enfermos, color=CEMP_PINK, lw=0.8, alpha=0.6)
            ax_calib.axvline(0.27, color=OPTIMAL_GREEN, linestyle="--", linewidth=1.5, label="Óptimo (0.27)")
            ax_calib.axvline(threshold, color=CEMP_DARK, linestyle="--", linewidth=2, label="Tu Selección")
            ax_calib.set_yticks([])
            ax_calib.set_xlim(-0.2, 1.25)
            ax_calib.spines['top'].set_visible(False)
            ax_calib.spines['right'].set_visible(False)
            ax_calib.spines['bottom'].set_visible(False)
            ax_calib.spines['left'].set_visible(False)
            ax_calib.set_xlabel("Probabilidad Predicha", fontsize=8, color="#888")
            ax_calib.legend(loc='upper right', fontsize=6, frameon=False)
            
            chart_html_calib = fig_to_html(fig_calib)
            st.markdown(f"""
            <div style="display:flex; justify-content:center; align-items:center; width:100%; height:100%;">
                {chart_html_calib}
            </div>
            """, unsafe_allow_html=True)
            plt.close(fig_calib)

    # LÓGICA IA
    input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
    prob = st.session_state.model.predict_proba(input_data)[0][1]
    is_high = prob > threshold 
    
    # CÁLCULO FIABILIDAD
    distancia_al_corte = abs(prob - threshold)
    if distancia_al_corte > 0.15:
        conf_text, conf_color = "ALTA", GOOD_TEAL
        conf_desc = "Probabilidad claramente alejada del umbral. Clasificación robusta."
    elif distancia_al_corte > 0.05:
        conf_text, conf_color = "MEDIA", "#F39C12"
        conf_desc = "Probabilidad relativamente cerca del umbral. Precaución."
    else:
        conf_text, conf_color = "BAJA", CEMP_PINK
        conf_desc = "Zona de incertidumbre clínica (Borderline). La probabilidad roza el umbral."

    # ESTILOS
    risk_color = CEMP_PINK if is_high else GOOD_TEAL
    risk_label = "ALTO RIESGO" if is_high else "BAJO RIESGO"
    risk_icon = "🔴" if is_high else "🟢"
    risk_bg = "#FFF5F5" if is_high else "#F0FDF4"
    risk_border = CEMP_PINK if is_high else GOOD_TEAL
    
    # ALERTAS
    alerts = []
    if glucose >= 200:
        alerts.append("Posible Diabetes (>200 mg/dL)")
    elif glucose >= 140:
        alerts.append("Intolerancia Glucosa / Prediabetes (140-199 mg/dL)")
    if bmi >= 40:
        alerts.append("Obesidad Mórbida (G3)")
    elif bmi >= 35:
        alerts.append("Obesidad G2")
    elif bmi >= 30:
        alerts.append("Obesidad G1")
    elif bmi >= 25:
        alerts.append("Sobrepeso")
    elif bmi < 18.5:
        alerts.append("Bajo Peso")
    if proxy_index > 19769.5: 
        alerts.append("Resistencia Insulina")
    
    if not alerts:
        insight_txt, insight_bd, alert_icon = "Sin hallazgos significativos", GOOD_TEAL, "✅"
    else:
        insight_txt, insight_bd, alert_icon = " • ".join(alerts), CEMP_PINK, "⚠️"

    # LAYOUT
    c_left, c_right = st.columns([1.8, 1], gap="medium") 
    
    # IZQUIERDA
    with c_left:
        # FICHA PACIENTE 
        st.markdown(f"""<div class="card card-auto" style="flex-direction:row; align-items:center; justify-content:space-between;">
            <div style="display:flex; align-items:center; gap:20px; flex-grow:1;">
                <div style="background:rgba(233, 127, 135, 0.1); width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2rem; color:{CEMP_DARK};">👤</div>
                <div>
                    <span class="card-header" style="margin-bottom:5px;">EXPEDIENTE MÉDICO</span>
                    <h2 style="margin:0; color:{CEMP_DARK}; font-size:1.6rem; line-height:1.2;">Paciente #8842-X</h2>
                    <div style="font-size:0.85rem; color:#666; margin-top:5px;">📅 Revisión: <b>14 Dic 2025</b></div>
                </div>
            </div>
            <div style="display:flex; flex-direction:column; align-items:center; gap:8px;">
                <div style="background:{risk_bg}; border:1px solid {risk_border}; color:{risk_border}; font-weight:bold; font-size:0.9rem; padding:8px 16px; border-radius:30px;">
                    {risk_icon} {risk_label}
                </div>
                <div style="background:#F8F9FA; border-radius:8px; padding: 4px 10px; border:1px solid #EEE;" title="{conf_desc}">
                    <span style="font-size:0.7rem; color:#999; font-weight:600;">FIABILIDAD: </span>
                    <span style="font-size:0.75rem; color:{conf_color}; font-weight:800;">{conf_text}</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        g_pos = min(100, max(0, (glucose - 50) / 3.0)) 
        b_pos = min(100, max(0, (bmi - 10) * 2.5)) 
        
        st.markdown(f"""<div class="card">
            <span class="card-header">CONTEXTO POBLACIONAL</span>
            <div style="margin-top:15px;">
                <div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">GLUCOSA 2H (TEST TOLERANCIA) <span style="font-weight:normal">({glucose} mg/dL)</span></div>
                <div class="bar-container">
                    <div class="bar-bg"><div class="bar-fill-glucose"></div></div>
                    <div class="bar-marker" style="left: {g_pos}%;"></div>
                    <div class="bar-txt" style="left: {g_pos}%;">{glucose}</div>
                </div>
                <div class="legend-container">
                    <span class="legend-label" style="left: 15%;">Normal (&lt;140)</span>
                    <span class="legend-label" style="left: 40%;">Prediabetes (140-199)</span>
                    <span class="legend-label" style="left: 75%;">Diabetes (&gt;200)</span>
                </div>
            </div>
            <div style="margin-top:35px;">
                <div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">ÍNDICE DE MASA CORPORAL <span style="font-weight:normal">({bmi:.1f})</span></div>
                <div class="bar-container">
                    <div class="bar-bg"><div class="bar-fill-bmi"></div></div>
                    <div class="bar-marker" style="left: {b_pos}%;"></div>
                    <div class="bar-txt" style="left: {b_pos}%;">{bmi:.1f}</div>
                </div>
                <div class="legend-container">
                    <span class="legend-label" style="left: 10%;">Bajo</span>
                    <span class="legend-label" style="left: 29%;">Normal</span>
                    <span class="legend-label" style="left: 43%;">Sobrepeso</span>
                    <span class="legend-label" style="left: 56%;">Ob. G1</span>
                    <span class="legend-label" style="left: 68%;">Ob. G2</span>
                    <span class="legend-label" style="left: 87%;">Ob. G3</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # DERECHA
    with c_right:
        st.markdown(f"""<div class="card card-auto" style="border-left:5px solid {insight_bd}; justify-content:center;">
            <span class="card-header" style="color:{insight_bd}; margin-bottom:10px;">HALLAZGOS CLAVE</span>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; color:{CEMP_DARK}; font-size:1.1rem; line-height:1.4;">{insight_txt}</h3>
                <div style="font-size:1.8rem;">{alert_icon}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.pie([prob, 1-prob], colors=[risk_color, '#F4F6F9'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15, edgecolor='none'))
        chart_html = fig_to_html(fig)
        plt.close(fig)

        prob_help = get_help_icon("Probabilidad calculada por el modelo de IA.")
        
        st.markdown(f"""<div class="card" style="text-align:center; justify-content: center;">
            <span class="card-header" style="justify-content:center; margin-bottom:15px;">PROBABILIDAD IA{prob_help}</span>
            <div style="position:relative; display:inline-block; margin: auto;">
                {chart_html}
                <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-size:2.5rem; font-weight:800; color:{CEMP_DARK}; letter-spacing:-1px;">
                    {prob*100:.1f}%
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

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

with tab3:
    st.write("")
    st.info("💡 Módulo de recomendaciones clínicas y generación de informes.")
