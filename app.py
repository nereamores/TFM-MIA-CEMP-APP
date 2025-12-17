import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import datetime

# =========================================================
# 1. CONFIGURACIÓN GLOBAL
# =========================================================
st.set_page_config(
    page_title="Diabetes NME",
    page_icon="🩸",
    layout="wide"
)

# =========================================================
# 2. GESTIÓN DE ESTADO
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "landing"

# Variable para controlar si se ha pulsado el botón "PREDECIR"
if "prediction_run" not in st.session_state:
    st.session_state.prediction_run = False

def ir_a_simulacion():
    st.session_state.page = "simulacion"

def volver_inicio():
    st.session_state.page = "landing"
    st.session_state.prediction_run = False # Reiniciar al salir

def ejecutar_prediccion():
    st.session_state.prediction_run = True

# =========================================================
# 3. PÁGINA: PORTADA
# =========================================================
if st.session_state.page == "landing":
    st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; }
        #MainMenu, footer, header { visibility: hidden; }
        
        .block-container {
            background-color: white; padding: 3rem !important; border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05); max-width: 800px !important;
            margin-top: 2rem; margin-left: auto !important; margin-right: auto !important;
        }
        
        /* FUENTE HELVETICA BOLD PARA LOGO */
        h1 {
            text-align: center; font-family: 'Helvetica', sans-serif !important;
            font-weight: 800 !important; font-size: 3.5rem !important;
            color: #2c3e50; margin-bottom: 0 !important; line-height: 1.2 !important;
            letter-spacing: -1px; cursor: default;
        }
        h1 a { display: none !important; pointer-events: none !important; }
        h1:hover { color: #2c3e50 !important; text-decoration: none !important; }
        
        .landing-pink { color: #ef7d86; }
        .landing-gray { color: #bdc3c7; }
        
        .badge-container { text-align: center; margin-bottom: 10px; }
        .badge {
            background-color: #2c3e50; color: white; padding: 6px 15px;
            border-radius: 50px; font-size: 11px; font-weight: bold;
            text-transform: uppercase; letter-spacing: 0.5px; display: inline-block;
        }
        .institution {
            text-align: center; color: #555; font-size: 13px; font-weight: 700;
            letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px;
        }
        .subtitle {
            text-align: center; font-size: 1.1rem; font-weight: 700;
            color: #34495e; margin-top: 5px; margin-bottom: 25px;
        }
        .description {
            text-align: center; color: #666; line-height: 1.6;
            font-size: 0.95rem; margin-bottom: 30px; padding: 0 20px;
        }
        .warning-box {
            background-color: #f9fafb; border-left: 4px solid #ef7d86;
            padding: 20px; border-radius: 4px; font-size: 0.85rem;
            color: #555; margin-bottom: 30px; text-align: center;
        }
        .warning-box p { margin: 0; line-height: 1.5; }
        
        div.stButton > button {
            position: relative; background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%);
            color: white; border: none; padding: 14px 80px; border-radius: 50px;
            font-weight: bold; font-size: 14px; text-transform: uppercase;
            letter-spacing: 1px; white-space: nowrap; box-shadow: 0 4px 15px rgba(239,112,122,0.3);
            cursor: pointer; overflow: visible;
        }
        div.stButton > button > span {
            position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%) translateX(4px);
            display: inline-block; pointer-events: none;
        }
        div.stButton > button::after {
            content: "➔"; position: absolute; right: 28px; top: 50%;
            transform: translateY(-50%); font-size: 16px; pointer-events: none;
        }
        div.stButton > button:hover {
            transform: translateY(-2px); box-shadow: 0 6px 20px rgba(239,112,122,0.5); color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    # HTML DE LA PORTADA SIN IDENTACIÓN PARA EVITAR ERRORES
    st.markdown("""
<div class="badge-container">
<span class="badge">TFM • Máster en Inteligencia Artificial aplicada a la salud</span>
</div>
<div class="institution">Centro Europeo de Másteres y Posgrados</div>
<h1>
    D<span class="landing-pink">IA</span>BETES<span class="landing-gray">.</span><span class="landing-pink">NME</span>
</h1>
<div class="subtitle">
    Prototipo de CDSS para el diagnóstico temprano de diabetes
</div>
<p class="description">
    Este proyecto explora el potencial de integrar modelos predictivos avanzados
    en el flujo de trabajo clínico, visualizando un futuro donde la IA actúa como
    un potente aliado en la detección temprana y prevención de la diabetes tipo 2.
</p>
<div class="warning-box">
    <p><strong>
        Aplicación desarrollada con fines exclusivamente educativos como parte de
        un Trabajo de Fin de Máster.
    </strong></p>
    <p style="margin-top:10px;">
        ⚠️ Esta herramienta NO es un dispositivo médico certificado.
        Los resultados son una simulación académica y NO deben utilizarse para el
        diagnóstico real.
    </p>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("INICIAR          "):
            ir_a_simulacion()
            st.rerun()

# =========================================================
# 4. PÁGINA: SIMULACIÓN
# =========================================================
elif st.session_state.page == "simulacion":

    CEMP_PINK = "#E97F87"
    CEMP_DARK = "#2C3E50" 
    GOOD_TEAL = "#4DB6AC"
    SLIDER_GRAY = "#BDC3C7"
    OPTIMAL_GREEN = "#8BC34A" 
    NOTE_GRAY_BG = "#F8F9FA" 
    NOTE_GRAY_TEXT = "#6C757D"

    BMI_GRADIENT = "linear-gradient(90deg, #81D4FA 0%, #4DB6AC 25%, #FFF176 40%, #FFB74D 55%, #E97F87 70%, #880E4F 100%)"
    GLUCOSE_GRADIENT = "linear-gradient(90deg, #4DB6AC 0%, #4DB6AC 28%, #FFF176 32%, #FFB74D 48%, #E97F87 52%, #880E4F 100%)"
    RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

    st.markdown(f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        .block-container {{
            background-color: transparent !important;
            box-shadow: none !important;
            max-width: 1250px !important; 
            padding-top: 2rem;
            padding-bottom: 2rem;
            margin: 0 auto;
        }}
        
        .cemp-logo {{ font-family: 'Helvetica', sans-serif; font-weight: 800; font-size: 1.8rem; color: {CEMP_DARK}; margin: 0; }}
        .cemp-logo span {{ color: {CEMP_PINK}; }}

        .stSlider {{ padding-top: 0px !important; padding-bottom: 10px !important; }}

        /* Estilos Tarjetas */
        .card-container {{
            background-color: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.04);
            margin-bottom: 15px; display: flex; flex-direction: column;
            justify-content: center; min-height: 300px;
        }}
        .card-auto {{ min-height: auto !important; height: 100%; }}
        
        .card-header {{
            color: #999; font-size: 0.75rem; font-weight: bold; letter-spacing: 1px;
            text-transform: uppercase; margin-bottom: 15px; display: flex; align-items: center;
        }}

        /* Barras */
        .bar-container {{ position: relative; width: 100%; margin-top: 20px; margin-bottom: 30px; }}
        .bar-bg {{ background: #F0F2F5; height: 12px; border-radius: 6px; width: 100%; overflow: hidden; }}
        .bar-fill-bmi {{ height: 100%; width: 100%; background: {BMI_GRADIENT}; border-radius: 6px; }}
        .bar-fill-glucose {{ height: 100%; width: 100%; background: {GLUCOSE_GRADIENT}; border-radius: 6px; }}
        .bar-marker {{ 
            position: absolute; top: -6px; width: 4px; height: 24px; 
            background: {CEMP_DARK}; border: 1px solid white; border-radius: 2px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3); transition: left 0.3s ease;
        }}
        .bar-txt {{ 
            position: absolute; top: -30px; transform: translateX(-50%); 
            font-size: 0.85rem; font-weight: bold; color: {CEMP_DARK}; 
            background: white; padding: 2px 8px; border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .legend-container {{ position: relative; width: 100%; height: 20px; margin-top: 8px; }}
        .legend-label {{
            position: absolute; transform: translateX(-50%); font-size: 0.7rem; 
            color: #888; font-weight: 600; text-align: center; white-space: nowrap;
        }}
        
        /* Botón Predecir Redondo y Rosa */
        div.stButton.predict-btn > button {{
            background: linear-gradient(90deg, #E97F87 0%, #ef707a 100%) !important;
            color: white !important;
            border-radius: 50px !important;
            height: 50px !important;
            width: 100% !important;
            font-weight: bold !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(233, 127, 135, 0.4) !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }}
        div.stButton.predict-btn > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(233, 127, 135, 0.6) !important;
        }}

        /* Sidebar Inputs */
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
            padding: 0px 5px; font-size: 0.9rem; text-align: center;
            color: {CEMP_DARK}; font-weight: 800; border-radius: 8px;
            background-color: white; border: 1px solid #ddd;
        }}
        .calc-box {{
            background-color: #F8F9FA; border-radius: 8px; padding: 12px 15px;
            border: 1px solid #EEE; margin-top: 5px; margin-bottom: 20px;
        }}
        .calc-label {{ font-size: 0.75rem; color: #888; font-weight: 600; text-transform: uppercase; }}
        .calc-value {{ font-size: 1rem; color: {CEMP_DARK}; font-weight: 800; }}
        </style>
    """, unsafe_allow_html=True)

    def fig_to_html(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=300)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        return f'<img src="data:image/png;base64,{img_str}" style="width:100%; object-fit:contain;">'

    def get_help_icon(description):
        return f"""<span style="display:inline-block; width:16px; height:16px; line-height:16px; text-align:center; border-radius:50%; background:#E0E0E0; color:#777; font-size:0.7rem; font-weight:bold; cursor:help; margin-left:6px; position:relative; top:-1px;" title="{description}">?</span>"""

    if 'model' not in st.session_state:
        class MockModel:
            def predict_proba(self, X):
                score = (X[0]*0.5) + (X[1]*0.4) + (X[3]*0.1) 
                prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
                return [[1-prob, prob]]
        st.session_state.model = MockModel()

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
                key=f"{key}_input", value=st.session_state[key], on_change=update_from_input, label_visibility="collapsed",
                # IMPORTANTE: Formato dinámico según tipo de dato
                format="%.2f" if isinstance(default_val, float) else "%d"
            )
        return st.session_state[key]

    with st.sidebar:
        if st.button("⬅ Volver"):
            volver_inicio()
            st.rerun()

        st.markdown(f'<div class="cemp-logo">D<span>IA</span>BETES<span style="color:{SLIDER_GRAY}">.</span><span>NME</span></div>', unsafe_allow_html=True)
        st.caption("CLINICAL DECISION SUPPORT SYSTEM")
        st.write("")
        
        # 1. METABÓLICOS (ENTEROS)
        glucose = input_biomarker("Glucosa 2h (mg/dL)", 50, 350, 120, "gluc", "Concentración plasmática a las 2h.")
        insulin = input_biomarker("Insulina (µU/ml)", 0, 900, 100, "ins", "Insulina a las 2h de ingesta.")
        
        proxy_index = glucose * insulin
        st.markdown(f"""
        <div class="calc-box" style="border-left: 4px solid {CEMP_PINK};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="calc-label">Índice RI (Glu x Ins)</span>
                <span class="calc-value">{proxy_index:.0f}</span> </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---") 

        # 2. ANTROPOMÉTRICOS (DECIMALES)
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
        pregnancies = input_biomarker("Embarazos", 0, 20, 1, "preg", "Nº veces embarazada.") 
        
        st.markdown("---") 

        # 4. DPF
        dpf = input_biomarker("Antecedentes (DPF)", 0.0, 2.5, 0.5, "dpf", "Estimación de predisposición genética.")

        if dpf <= 0.15: dpf_label, bar_color = "Carga muy baja", GOOD_TEAL
        elif dpf <= 0.40: dpf_label, bar_color = "Carga baja", "#D4E157"
        elif dpf <= 0.80: dpf_label, bar_color = "Carga moderada", "#FFB74D"
        elif dpf <= 1.20: dpf_label, bar_color = "Carga elevada", CEMP_PINK
        else: dpf_label, bar_color = "Carga muy elevada", "#880E4F"

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:-10px; margin-bottom:2px;">
            <span style="font-size:0.8rem; font-weight:bold; color:{bar_color};">{dpf_label}</span>
            <span style="font-size:0.8rem; color:#666;">{dpf:.2f}</span>
        </div>
        <div style="width:100%; background-color:#F0F2F5; border-radius:4px; height:8px; margin-bottom:10px;">
            <div style="width:{min(100, (dpf/2.5)*100)}%; background-color:{bar_color}; height:8px; border-radius:4px; transition: width 0.3s ease;"></div>
        </div>
        """, unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    st.markdown(f"<h1 style='color:{CEMP_DARK}; margin-bottom: 10px; font-size: 2.2rem;'>Evaluación de Riesgo Diabético</h1>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Panel General", "Factores (SHAP)", "Protocolo"])

    with tab1:
        st.write("")
        
        # LOGICA (Cálculo)
        input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
        prob = st.session_state.model.predict_proba(input_data)[0][1]
        
        with st.expander("Ajuste de Sensibilidad Clínica"):
            st.caption("Selecciona manualmente el umbral de decisión.")
            threshold = st.slider("Umbral", 0.0, 1.0, 0.27, 0.01, label_visibility="collapsed")
        
        is_high = prob > threshold
        risk_color = CEMP_PINK if is_high else GOOD_TEAL
        risk_label = "ALTO RIESGO" if is_high else "BAJO RIESGO"
        risk_icon = "🔴" if is_high else "🟢"
        risk_bg = "#FFF5F5" if is_high else "#F0FDF4"
        risk_border = CEMP_PINK if is_high else GOOD_TEAL

        # --- LAYOUT DOS COLUMNAS ---
        c_left, c_right = st.columns([1.8, 1], gap="medium") 
        
        with c_left:
            # --- TARJETA EXPEDIENTE (Input de usuario incrustado) ---
            with st.container():
                st.markdown("""<div class="card-container">""", unsafe_allow_html=True)
                
                # Header Tarjeta
                st.markdown(f"""
                    <div style="display:flex; justify-content: space-between; align-items: flex-start;">
                        <div style="display:flex; align-items:center; gap:15px; margin-bottom:15px; width: 100%;">
                            <div style="background:rgba(233, 127, 135, 0.1); width:50px; height:50px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.5rem; color:{CEMP_DARK}; flex-shrink: 0;">👤</div>
                            <div style="width: 100%;">
                                <span style="color:#999; font-size:0.75rem; font-weight:bold; letter-spacing:1px; text-transform:uppercase;">EXPEDIENTE MÉDICO</span>
                            </div>
                        </div>
                """, unsafe_allow_html=True)
                
                # Badge de Riesgo (SOLO SI PREDICT = TRUE)
                if st.session_state.prediction_run:
                    st.markdown(f"""
                        <div style="background:{risk_bg}; border:1px solid {risk_border}; color:{risk_border}; font-weight:bold; font-size:0.8rem; padding:5px 12px; border-radius:20px; white-space: nowrap;">
                            {risk_icon} {risk_label}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True) # Cierre flex header

                # INPUTS DE TEXTO Y FECHA
                col_n, col_d = st.columns([2, 1])
                with col_n:
                    st.text_input("Nombre", value="Paciente #8842-X", label_visibility="collapsed", placeholder="Nombre del paciente")
                with col_d:
                    st.date_input("Fecha", datetime.date.today(), label_visibility="collapsed")
                
                st.markdown("</div>", unsafe_allow_html=True) # Cierre tarjeta

            # --- CONTEXTO POBLACIONAL ---
            g_pos = min(100, max(0, (glucose - 50) / 3.0)) 
            b_pos = min(100, max(0, (bmi - 10) * 2.5)) 
            st.markdown(f"""<div class="card-container">
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
                        <span class="legend-label" style="left: 40%;">Intolerancia (140-199)</span>
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

        with c_right:
            # --- HALLAZGOS CLAVE (Solo visible tras predecir) ---
            if st.session_state.prediction_run:
                # Logica Alertas
                alerts = []
                if glucose >= 200: alerts.append("Posible Diabetes")
                elif glucose >= 140: alerts.append("Posible Prediabetes")
                if bmi >= 40: alerts.append("Obesidad G3")
                elif bmi >= 30: alerts.append("Obesidad")
                if proxy_index > 19769.5: alerts.append("Resistencia Insulina")
                
                if not alerts:
                    insight_txt, alert_icon = "Sin hallazgos significativos", "✅"
                else:
                    insight_txt, alert_icon = " • ".join(alerts), "⚠️"

                st.markdown(f"""<div class="card-container card-auto" style="border-left:5px solid {insight_bd if 'insight_bd' in locals() else CEMP_PINK}; justify-content:center;">
                    <span class="card-header" style="color:{CEMP_PINK}; margin-bottom:10px;">HALLAZGOS CLAVE</span>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0; color:{CEMP_DARK}; font-size:1.1rem; line-height:1.4;">{insight_txt}</h3>
                        <div style="font-size:1.8rem;">{alert_icon}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                # Espacio vacío para mantener estructura
                st.markdown(f"""<div class="card-container card-auto" style="border:1px dashed #eee; justify-content:center; align-items:center;">
                    <span style="color:#ccc;">Análisis pendiente...</span>
                </div>""", unsafe_allow_html=True)

            # --- PROBABILIDAD IA (Con Botón o Gráfico) ---
            st.markdown(f"""<div class="card-container" style="text-align:center; justify-content: center;">
                <span class="card-header" style="justify-content:center; margin-bottom:15px;">PROBABILIDAD IA</span>""", unsafe_allow_html=True)
            
            if not st.session_state.prediction_run:
                # BOTÓN PREDECIR
                st.markdown('<div style="margin: 20px 0;">', unsafe_allow_html=True)
                # Clase CSS 'predict-btn' definida arriba para estilo rosa y redondo
                if st.button("PREDECIR RIESGO", key="btn_predict", type="primary"):
                    ejecutar_prediccion()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # GRÁFICO
                fig, ax = plt.subplots(figsize=(3.2, 3.2))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                ax.pie([prob, 1-prob], colors=[risk_color, '#F4F6F9'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15, edgecolor='none'))
                threshold_angle = 90 - (threshold * 360)
                theta_rad = np.deg2rad(threshold_angle)
                x1 = 0.85 * np.cos(theta_rad)
                y1 = 0.85 * np.sin(theta_rad)
                x2 = 1.15 * np.cos(theta_rad)
                y2 = 1.15 * np.sin(theta_rad)
                ax.plot([x1, x2], [y1, y2], color=CEMP_DARK, linestyle='--', linewidth=2)
                chart_html = fig_to_html(fig)
                plt.close(fig)

                st.markdown(f"""
                <div style="position:relative; display:inline-block; margin: auto;">
                    {chart_html}
                    <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-size:2.5rem; font-weight:800; color:{CEMP_DARK}; letter-spacing:-1px;">
                        {prob*100:.1f}%
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 0.65rem; color: #999; display: flex; align-items: center; justify-content: center; gap: 5px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase;">
                    <span style="display: inline-block; width: 15px; border-top: 2px dashed {CEMP_DARK};"></span>
                    <span>Umbral de decisión</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        if st.session_state.prediction_run:
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
            st.markdown(f"""<div class="card-container">
            <h3 style="color:{CEMP_DARK}; font-size:1.2rem; margin-bottom:5px;">Factores de Riesgo (SHAP)</h3>
            <span class="card-header" style="margin-bottom:20px;">EXPLICABILIDAD DEL MODELO</span>
            {chart_html}
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Ejecute la predicción para ver el análisis de factores.")

    with tab3:
        st.write("")
        st.info("💡 Módulo de recomendaciones clínicas y generación de informes.")
