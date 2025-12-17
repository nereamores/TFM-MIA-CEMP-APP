import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# =========================================================
# 1. CONFIGURACIÓN INICIAL (GLOBAL)
# =========================================================
# Usamos "wide" para que la simulación se vea bien.
# En la portada usaremos columnas para centrarlo.
st.set_page_config(
    page_title="Diabetes NME",
    page_icon="🩸",
    layout="wide"
)

# --- ESTADO DE NAVEGACIÓN ---
if "page" not in st.session_state:
    st.session_state.page = "landing"

def ir_a_simulacion():
    st.session_state.page = "simulacion"

def volver_inicio():
    st.session_state.page = "landing"

# =========================================================
# 2. PÁGINA: PORTADA (LANDING) - TU DISEÑO ORIGINAL
# =========================================================
if st.session_state.page == "landing":

    # CSS GLOBAL DE TU PORTADA (Centrado y estilos)
    st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        #MainMenu, footer, header {
            visibility: hidden;
        }
        
        /* Título */
        .big-title {
            text-align: center;
            font-family: Arial, sans-serif;
            font-weight: 900 !important;
            font-size: 3.5rem !important;
            color: #2c3e50;
            margin-bottom: 0 !important;
            line-height: 1.2 !important;
        }
        .landing-pink { color: #ef7d86; }
        .landing-gray { color: #bdc3c7; }

        /* Badges */
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

        /* Textos */
        .subtitle {
            text-align: center; font-size: 1.1rem; font-weight: 700;
            color: #34495e; margin-top: 5px; margin-bottom: 25px;
        }
        .description {
            text-align: center; color: #666; line-height: 1.6;
            font-size: 0.95rem; margin-bottom: 30px; padding: 0 20px;
        }

        /* Warning Box */
        .warning-box {
            background-color: #f9fafb; border-left: 4px solid #ef7d86;
            padding: 20px; border-radius: 4px; font-size: 0.85rem;
            color: #555; margin-bottom: 30px; text-align: center;
        }
        .warning-box p { margin: 0; line-height: 1.5; }

        /* --- BOTÓN ORIGINAL --- */
        div.stButton > button {
            position: relative;
            background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%);
            color: white; border: none; padding: 14px 80px;
            border-radius: 50px; font-weight: bold; font-size: 14px;
            text-transform: uppercase; letter-spacing: 1px; white-space: nowrap;
            box-shadow: 0 4px 15px rgba(239,112,122,0.3); cursor: pointer;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(239,112,122,0.5); color: white;
        }
        /* Flecha decorativa CSS */
        div.stButton > button::after {
            content: "➔"; position: absolute; right: 28px; top: 50%;
            transform: translateY(-50%); font-size: 16px; pointer-events: none;
        }
    </style>
    """, unsafe_allow_html=True)

    # USAMOS COLUMNAS PARA CENTRAR EL CONTENIDO (porque estamos en modo 'wide')
    # Esto simula tu "block-container" centrado
    col_izq, col_centro, col_der = st.columns([1, 2, 1])

    with col_centro:
        # Renderizamos el HTML de la portada dentro de la columna central
        st.markdown("""
            <div style="background-color: white; padding: 3rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05);">
                <div class="badge-container">
                    <span class="badge">TFM • Máster en Inteligencia Artificial aplicada a la salud</span>
                </div>

                <div class="institution">Centro Europeo de Másteres y Posgrados</div>

                <h1 class="big-title">
                    D<span class="landing-pink">IA</span>BETES
                    <span class="landing-gray">.</span>
                    <span class="landing-pink">NME</span>
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
            </div>
        """, unsafe_allow_html=True)

        st.write("") # Espacio separador

        # BOTÓN CENTRADO (usando columnas anidadas)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            # Tu botón original con espacios para que salga ancho
            if st.button("INICIAR          "):
                ir_a_simulacion()
                st.rerun()

# =========================================================
# 3. PÁGINA: SIMULACIÓN (DASHBOARD AVANZADO)
# =========================================================
elif st.session_state.page == "simulacion":

    # --- COLORES Y GRADIENTES ---
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

    # --- CSS DASHBOARD ---
    st.markdown(f"""
        <style>
        /* Limpiamos estilos anteriores y aplicamos los del dashboard */
        .block-container {{ max-width: 1250px; padding-top: 1rem; padding-bottom: 2rem; margin: 0 auto; }}
        
        /* LOGO */
        .cemp-logo {{ font-family: 'Helvetica', sans-serif; font-weight: 800; font-size: 1.8rem; color: {CEMP_DARK}; margin: 0; }}
        .cemp-logo span {{ color: {CEMP_PINK}; }}

        /* SLIDER Y INPUTS */
        .stSlider {{ padding-top: 0px !important; padding-bottom: 10px !important; }}
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
            padding: 0px 5px; font-size: 0.9rem; text-align: center; color: {CEMP_DARK}; font-weight: 800; border-radius: 8px; background-color: white; border: 1px solid #ddd;
        }}

        /* CAJAS Y TARJETAS */
        .calc-box {{ background-color: #F8F9FA; border-radius: 8px; padding: 12px 15px; border: 1px solid #EEE; margin-top: 5px; margin-bottom: 20px; }}
        .calc-label {{ font-size: 0.75rem; color: #888; font-weight: 600; text-transform: uppercase; }}
        .calc-value {{ font-size: 1rem; color: {CEMP_DARK}; font-weight: 800; }}
        
        .card {{ background-color: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.04); margin-bottom: 15px; display: flex; flex-direction: column; justify-content: center; min-height: 300px; }}
        .card-auto {{ min-height: auto !important; height: 100%; }}
        .card-header {{ color: #999; font-size: 0.75rem; font-weight: bold; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 15px; display: flex; align-items: center; }}

        /* GRÁFICOS DE BARRAS HTML */
        .bar-container {{ position: relative; width: 100%; margin-top: 20px; margin-bottom: 30px; }}
        .bar-bg {{ background: #F0F2F5; height: 12px; border-radius: 6px; width: 100%; overflow: hidden; }}
        .bar-fill-bmi {{ height: 100%; width: 100%; background: {BMI_GRADIENT}; border-radius: 6px; }}
        .bar-fill-glucose {{ height: 100%; width: 100%; background: {GLUCOSE_GRADIENT}; border-radius: 6px; }}
        .bar-marker {{ position: absolute; top: -6px; width: 4px; height: 24px; background: {CEMP_DARK}; border: 1px solid white; border-radius: 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); transition: left 0.3s ease; }}
        .bar-txt {{ position: absolute; top: -30px; transform: translateX(-50%); font-size: 0.85rem; font-weight: bold; color: {CEMP_DARK}; background: white; padding: 2px 8px; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .legend-container {{ position: relative; width: 100%; height: 20px; margin-top: 8px; }}
        .legend-label {{ position: absolute; transform: translateX(-50%); font-size: 0.7rem; color: #888; font-weight: 600; white-space: nowrap; }}
        </style>
    """, unsafe_allow_html=True)

    # --- HELPERS ---
    def fig_to_html(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=300)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        return f'<img src="data:image/png;base64,{img_str}" style="width:100%; object-fit:contain;">'

    def get_help_icon(description):
        return f"""<span style="display:inline-block; width:16px; height:16px; line-height:16px; text-align:center; border-radius:50%; background:#E0E0E0; color:#777; font-size:0.7rem; font-weight:bold; cursor:help; margin-left:6px; position:relative; top:-1px;" title="{description}">?</span>"""

    # --- MODELO MOCK ---
    if 'model' not in st.session_state:
        class MockModel:
            def predict_proba(self, X):
                score = (X[0]*0.5) + (X[1]*0.4) + (X[3]*0.1) 
                prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
                return [[1-prob, prob]]
        st.session_state.model = MockModel()

    # --- INPUTS SINCRONIZADOS ---
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
            st.slider(label="", min_value=min_val, max_value=max_val, step=step, key=f"{key}_slider", value=st.session_state[key], on_change=update_from_slider, label_visibility="collapsed")
        with c2:
            st.number_input(label="", min_value=min_val, max_value=max_val, step=step, key=f"{key}_input", value=st.session_state[key], on_change=update_from_input, label_visibility="collapsed")
        return st.session_state[key]

    # --- BARRA LATERAL (SIDEBAR) ---
    with st.sidebar:
        if st.button("⬅ Volver"):
            volver_inicio()
            st.rerun()

        st.markdown(f'<div class="cemp-logo">D<span>IA</span>BETES<span style="color:{SLIDER_GRAY}">.</span><span>NME</span></div>', unsafe_allow_html=True)
        st.caption("CLINICAL DECISION SUPPORT SYSTEM")
        st.write("")
        
        # 1. METABÓLICOS
        glucose = input_biomarker("Glucosa 2h (mg/dL)", 50, 350, 120, "gluc", "Concentración plasmática a las 2h.")
        insulin = input_biomarker("Insulina (µU/ml)", 0, 900, 100, "ins", "Insulina a las 2h de ingesta.")
        
        proxy_index = glucose * insulin
        st.markdown(f"""
        <div class="calc-box" style="border-left: 4px solid {CEMP_PINK};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="calc-label">Índice RI</span>
                <span class="calc-value">{proxy_index:,.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---") 

        # 2. ANTROPOMÉTRICOS
        weight = input_biomarker("Peso (kg)", 30.0, 250.0, 70.0, "weight")
        height = input_biomarker("Altura (m)", 1.00, 2.20, 1.70, "height")
        
        bmi = weight / (height * height)
        
        st.markdown(f"""
        <div class="calc-box" style="border-left: 4px solid {CEMP_PINK};">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span class="calc-label">BMI (kg/m²)</span>
                <span class="calc-value">{bmi:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---") 

        # 3. PACIENTE (AQUÍ ESTABA EL ERROR, YA CORREGIDO)
        c_age, c_preg = st.columns(2)
        age = input_biomarker("Edad (años)", 18, 90, 45, "age")
        pregnancies = input_biomarker("Embarazos", 0, 20, 1, "preg", "Nº veces embarazada.") 
        
        st.markdown("---") 

        # 4. DPF
        dpf = input_biomarker("Antecedentes (DPF)", 0.0, 2.5, 0.5, "dpf", "Estimación de predisposición genética.")

    # --- MAIN CONTENT DASHBOARD ---
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
                <div style="background-color:{NOTE_GRAY_BG}; padding:15px; border-radius:8px; border:1px solid #E9ECEF; color:{NOTE_GRAY_TEXT}; font-size:0.85rem;">
                    <strong>Criterio Técnico:</strong> Se ha seleccionado <strong>0.27</strong> como umbral óptimo.
                </div>
                """, unsafe_allow_html=True)

            with c_calib_2:
                # --- SIMULACIÓN MATEMÁTICA ---
                x = np.linspace(-0.15, 1.25, 500)
                y_sanos = 1.9 * np.exp(-((x - 0.1)**2) / (2 * 0.11**2)) + 0.5 * np.exp(-((x - 0.55)**2) / (2 * 0.15**2))
                y_enfermos = 0.35 * np.exp(-((x - 0.28)**2) / (2 * 0.1**2)) + 1.4 * np.exp(-((x - 0.68)**2) / (2 * 0.16**2))
                
                fig_calib, ax_calib = plt.subplots(figsize=(6, 2.5))
                fig_calib.patch.set_facecolor('none')
                ax_calib.set_facecolor('none')
                ax_calib.fill_between(x, y_sanos, color="#BDC3C7", alpha=0.3)
                ax_calib.plot(x, y_sanos, color="gray", lw=0.8, alpha=0.6)
                ax_calib.fill_between(x, y_enfermos, color=CEMP_PINK, alpha=0.3)
                ax_calib.plot(x, y_enfermos, color=CEMP_PINK, lw=0.8, alpha=0.6)
                ax_calib.axvline(0.27, color=OPTIMAL_GREEN, linestyle="--", linewidth=1.5)
                ax_calib.axvline(threshold, color=CEMP_DARK, linestyle="--", linewidth=2)
                ax_calib.set_yticks([])
                ax_calib.set_xlim(-0.2, 1.25)
                ax_calib.axis('off')
                
                chart_html_calib = fig_to_html(fig_calib)
                st.markdown(f"""<div style="display:flex; justify-content:center;">{chart_html_calib}</div>""", unsafe_allow_html=True)
                plt.close(fig_calib)

        # LÓGICA IA
        input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
        prob = st.session_state.model.predict_proba(input_data)[0][1]
        is_high = prob > threshold 
        
        # ESTILOS RESULTADO
        risk_label = "ALTO RIESGO" if is_high else "BAJO RIESGO"
        risk_icon = "🔴" if is_high else "🟢"
        risk_bg = "#FFF5F5" if is_high else "#F0FDF4"
        risk_border = CEMP_PINK if is_high else GOOD_TEAL
        
        # LAYOUT RESULTADOS
        c_left, c_right = st.columns([1.8, 1], gap="medium") 
        
        with c_left:
            st.markdown(f"""<div class="card card-auto" style="flex-direction:row; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:20px; flex-grow:1;">
                    <div style="background:rgba(233, 127, 135, 0.1); width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2rem; color:{CEMP_DARK};">👤</div>
                    <div>
                        <span class="card-header" style="margin-bottom:5px;">EXPEDIENTE MÉDICO</span>
                        <h2 style="margin:0; color:{CEMP_DARK}; font-size:1.6rem; line-height:1.2;">Paciente #8842-X</h2>
                    </div>
                </div>
                <div style="background:{risk_bg}; border:1px solid {risk_border}; color:{risk_border}; font-weight:bold; font-size:0.9rem; padding:8px 16px; border-radius:30px;">
                    {risk_icon} {risk_label}
                </div>
            </div>""", unsafe_allow_html=True)

            g_pos = min(100, max(0, (glucose - 50) / 3.0)) 
            b_pos = min(100, max(0, (bmi - 10) * 2.5)) 
            
            st.markdown(f"""<div class="card">
                <span class="card-header">CONTEXTO POBLACIONAL</span>
                <div style="margin-top:15px;">
                    <div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">GLUCOSA <span style="font-weight:normal">({glucose} mg/dL)</span></div>
                    <div class="bar-container">
                        <div class="bar-bg"><div class="bar-fill-glucose"></div></div>
                        <div class="bar-marker" style="left: {g_pos}%;"></div>
                    </div>
                </div>
                <div style="margin-top:35px;">
                    <div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">IMC <span style="font-weight:normal">({bmi:.1f})</span></div>
                    <div class="bar-container">
                        <div class="bar-bg"><div class="bar-fill-bmi"></div></div>
                        <div class="bar-marker" style="left: {b_pos}%;"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        with c_right:
            # DONUT CHART
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            ax.pie([prob, 1-prob], colors=[CEMP_PINK if is_high else GOOD_TEAL, '#F4F6F9'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15, edgecolor='none'))
            
            chart_html = fig_to_html(fig)
            plt.close(fig)
            
            st.markdown(f"""<div class="card" style="text-align:center; justify-content: center;">
                <span class="card-header" style="justify-content:center; margin-bottom:15px;">PROBABILIDAD IA</span>
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
        ax.axis('off')
        chart_html = fig_to_html(fig)
        plt.close(fig)
        st.markdown(f"""<div class="card">
        <h3 style="color:{CEMP_DARK}; font-size:1.2rem; margin-bottom:5px;">Factores de Riesgo (SHAP)</h3>
        {chart_html}
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.info("💡 Módulo de recomendaciones clínicas.")
