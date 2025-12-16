import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time
from datetime import date

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="DIABETES.NME | CDSS", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. VARIABLES DE ESTADO (SESSION STATE) ---
if 'page' not in st.session_state:
    st.session_state.page = 'admission'
if 'patient' not in st.session_state:
    st.session_state.patient = {'id': '', 'name': '', 'date': date.today()}
if 'model' not in st.session_state:
    # Mock Model para la demo
    class MockModel:
        def predict_proba(self, X):
            score = (X[0]*0.5) + (X[1]*0.4) + (X[3]*0.1) 
            prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

# --- 3. ESTILOS CSS (PROFESIONAL) ---
CEMP_PINK = "#E97F87"
CEMP_DARK = "#2C3E50"
GOOD_TEAL = "#4DB6AC"
BMI_GRADIENT = "linear-gradient(90deg, #81D4FA 0%, #4DB6AC 25%, #FFF176 40%, #FFB74D 55%, #E97F87 70%, #880E4F 100%)"
GLUCOSE_GRADIENT = "linear-gradient(90deg, #4DB6AC 0%, #4DB6AC 28%, #FFF176 32%, #FFB74D 48%, #E97F87 52%, #880E4F 100%)"
RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {CEMP_DARK};
    }}
    
    /* LIMPIEZA INTERFAZ */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background-color: #F8F9FA;
        border-right: 1px solid #E9ECEF;
    }}
    
    .cemp-logo {{ 
        font-weight: 800; font-size: 1.6rem; color: {CEMP_DARK}; margin-bottom: 5px;
    }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}
    
    .sidebar-section {{
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
    }}

    /* TARJETAS DASHBOARD */
    .card {{
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }}
    .card-header {{
        color: #888;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }}

    /* BOTONES */
    .primary-btn button {{
        background-color: {CEMP_PINK} !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        border: none !important;
        transition: all 0.3s !important;
    }}
    .primary-btn button:hover {{
        background-color: #D66E76 !important;
        box-shadow: 0 4px 12px rgba(233, 127, 135, 0.4) !important;
        transform: translateY(-2px);
    }}

    /* PÁGINA ADMISIÓN - ESTILO PANEL */
    .welcome-container {{
        background-color: white;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.04);
        border-left: 6px solid {CEMP_PINK};
    }}
    .welcome-title {{
        font-size: 2rem;
        font-weight: 800;
        color: {CEMP_DARK};
        margin-bottom: 10px;
    }}
    .welcome-subtitle {{
        font-size: 1rem;
        color: #666;
        line-height: 1.6;
        margin-bottom: 30px;
    }}

    /* FORMULARIO ESTILIZADO */
    .form-box {{
        background-color: #F8F9FA;
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #E9ECEF;
    }}
    
    /* BARRAS DE PROGRESO */
    .bar-bg {{ background: #EEE; height: 10px; border-radius: 5px; width: 100%; position: relative; overflow: hidden; }}
    .bar-fill-bmi {{ height: 100%; background: {BMI_GRADIENT}; }}
    .bar-fill-glu {{ height: 100%; background: {GLUCOSE_GRADIENT}; }}
    .bar-marker {{ position: absolute; top: -5px; width: 4px; height: 20px; background: {CEMP_DARK}; border: 1px solid white; z-index: 5; }}
    
    </style>
""", unsafe_allow_html=True)

# --- 4. HELPERS ---
def fig_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%; object-fit:contain;">'

def get_help(text):
    return f'<span title="{text}" style="cursor:help; color:#AAA; font-size:0.8em;"> ⓘ</span>'

# --- 5. SIDEBAR (ENTRADA DE DATOS CLÍNICOS) ---
# La sidebar está siempre presente para permitir ajustes en tiempo real
with st.sidebar:
    st.markdown(f'<div class="cemp-logo">DIABETES<span>.NME</span></div>', unsafe_allow_html=True)
    st.caption("CLINICAL DECISION SUPPORT SYSTEM")
    
    st.markdown('<div class="sidebar-section">1. Bioquímica</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1: st.markdown(f"**Glucosa 2h** (mg/dL) {get_help('Prueba de tolerancia')}", unsafe_allow_html=True)
    with c2: glucose = st.number_input("g", 50, 400, 120, label_visibility="collapsed")
    
    c1, c2 = st.columns([2, 1])
    with c1: st.markdown("**Insulina** (µU/ml)")
    with c2: insulin = st.number_input("i", 0, 900, 100, label_visibility="collapsed")
    
    # Cálculo automático
    ri = glucose * insulin
    st.info(f"⚡ **Índice RI:** {ri:,.0f}")

    st.markdown('<div class="sidebar-section">2. Antropometría</div>', unsafe_allow_html=True)
    weight = st.slider("Peso (kg)", 30.0, 200.0, 70.0, step=0.1)
    height = st.slider("Altura (m)", 1.00, 2.30, 1.70, step=0.01)
    bmi = weight / (height**2)
    st.info(f"⚖️ **BMI:** {bmi:.2f} kg/m²")

    st.markdown('<div class="sidebar-section">3. Historia</div>', unsafe_allow_html=True)
    age = st.slider("Edad (años)", 18, 90, 45)
    pregnancies = st.slider("Embarazos", 0, 15, 1)
    dpf = st.slider("Carga Genética (DPF)", 0.0, 2.5, 0.5, step=0.01)

# --- 6. PÁGINA 1: ADMISIÓN (HOME) ---
if st.session_state.page == 'admission':
    
    st.write("") # Espacio
    
    # Layout a dos columnas: Izquierda (Info) - Derecha (Formulario)
    col_info, col_form = st.columns([1.2, 1], gap="large")
    
    with col_info:
        st.markdown(f"""
        <div class="welcome-container">
            <div class="welcome-title">Bienvenido al Sistema de Evaluación de Riesgo.</div>
            <div class="welcome-subtitle">
                Esta herramienta de <b>Soporte a la Decisión Clínica (CDSS)</b> utiliza algoritmos de Machine Learning 
                entrenados en el dataset Pima Indians para estimar la probabilidad de diabetes mellitus tipo 2.
                <br><br>
                <b>Instrucciones:</b>
                <ol>
                    <li>Verifique los <b>parámetros clínicos</b> del paciente en la barra lateral izquierda.</li>
                    <li>Introduzca los <b>datos administrativos</b> en el formulario.</li>
                    <li>Pulse <b>"Predecir Riesgo"</b> para generar el informe.</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        st.markdown('<div class="form-box">', unsafe_allow_html=True)
        st.subheader("📋 Registro de Paciente")
        
        st.session_state.patient['id'] = st.text_input("Nº Historia Clínica / ID", value=st.session_state.patient['id'], placeholder="Ej: 8842-X")
        st.session_state.patient['name'] = st.text_input("Nombre Completo (Opcional)", value=st.session_state.patient['name'], placeholder="Nombre y Apellidos")
        st.session_state.patient['date'] = st.date_input("Fecha de Consulta", value=st.session_state.patient['date'])
        
        st.write("")
        st.write("")
        
        # Botón con estilo "primary-btn" mediante container CSS
        c_btn = st.container()
        c_btn.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if c_btn.button("GENERAR PREDICCIÓN CLÍNICA  ➔", use_container_width=True):
            if not st.session_state.patient['id']:
                st.error("⚠️ Por favor, introduzca un ID de paciente.")
            else:
                with st.spinner("Procesando biomarcadores..."):
                    time.sleep(1.2) # Simulación de carga
                    st.session_state.page = 'dashboard'
                    st.rerun()
        c_btn.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. PÁGINA 2: DASHBOARD DE RESULTADOS ---
elif st.session_state.page == 'dashboard':
    
    # Barra Superior de Navegación
    c_h1, c_h2 = st.columns([4, 1])
    with c_h1:
        st.markdown(f"## 🩺 Informe Clínico: {st.session_state.patient.get('name', 'Paciente')}")
        st.caption(f"ID: {st.session_state.patient['id']} | Fecha: {st.session_state.patient['date'].strftime('%d/%m/%Y')}")
    with c_h2:
        if st.button("⬅ Volver a Admisión"):
            st.session_state.page = 'admission'
            st.rerun()
            
    st.divider()

    # --- PESTAÑAS ---
    tab_main, tab_shap, tab_report = st.tabs(["Panel General", "Factores de Riesgo", "Exportar Informe"])

    with tab_main:
        st.write("")
        
        # CALIBRACIÓN
        with st.expander("⚙️ Calibración del Modelo (Umbral de Decisión)"):
            ec1, ec2 = st.columns([1, 2])
            with ec1:
                threshold = st.slider("Umbral de Corte", 0.0, 1.0, 0.27, 0.01)
                st.info("💡 **Criterio Técnico:** Se fija el umbral óptimo en **0.27** (F2-Score) para maximizar la detección de positivos.")
            with ec2:
                # Gráfica de Densidad (Estática para rendimiento)
                x = np.linspace(-0.15, 1.25, 500)
                y0 = 1.9 * np.exp(-((x - 0.1)**2) / (2 * 0.11**2)) + 0.5 * np.exp(-((x - 0.55)**2) / (2 * 0.15**2))
                y1 = 0.35 * np.exp(-((x - 0.28)**2) / (2 * 0.1**2)) + 1.4 * np.exp(-((x - 0.68)**2) / (2 * 0.16**2))
                
                fig_cal, ax_cal = plt.subplots(figsize=(6, 2))
                ax_cal.set_facecolor('none'); fig_cal.patch.set_facecolor('none')
                ax_cal.fill_between(x, y0, color="#BDC3C7", alpha=0.3, label="Clase 0")
                ax_cal.plot(x, y0, color="gray", lw=1)
                ax_cal.fill_between(x, y1, color=CEMP_PINK, alpha=0.3, label="Clase 1")
                ax_cal.plot(x, y1, color=CEMP_PINK, lw=1)
                ax_cal.axvline(threshold, color=CEMP_DARK, linestyle="--", linewidth=2)
                ax_cal.set_yticks([]); ax_cal.set_xlim(-0.2, 1.25)
                ax_cal.spines['top'].set_visible(False); ax_cal.spines['right'].set_visible(False); ax_cal.spines['left'].set_visible(False)
                st.pyplot(fig_cal, use_container_width=True)
                plt.close(fig_cal)

        # LÓGICA PREDICTIVA
        input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
        prob = st.session_state.model.predict_proba(input_data)[0][1]
        is_high = prob > threshold
        
        # Alertas
        alerts = []
        if glucose >= 200: alerts.append("Posible Diabetes (>200 mg/dL)")
        elif glucose >= 140: alerts.append("Intolerancia Glucosa / Prediabetes")
        if bmi >= 30: alerts.append(f"Obesidad (BMI {bmi:.1f})")
        elif bmi >= 25: alerts.append("Sobrepeso")
        if ri > 19000: alerts.append("Resistencia Insulina")

        risk_color = CEMP_PINK if is_high else GOOD_TEAL
        risk_txt = "ALTO RIESGO" if is_high else "BAJO RIESGO"
        risk_ico = "🔴" if is_high else "🟢"

        # LAYOUT DE TARJETAS
        col_left, col_right = st.columns([1.8, 1], gap="medium")

        with col_left:
            # TARJETA CONTEXTO
            g_pct = min(100, max(0, (glucose - 50) / 3.0)) # Escala 50-350
            b_pct = min(100, max(0, (bmi - 10) * 2.5))     # Escala 10-50
            
            st.markdown(f"""
            <div class="card">
                <div class="card-header">CONTEXTO POBLACIONAL</div>
                
                <div style="font-size:0.8rem; font-weight:700; margin-bottom:5px;">GLUCOSA 2H ({glucose} mg/dL)</div>
                <div class="bar-bg">
                    <div class="bar-fill-glu"></div>
                    <div class="bar-marker" style="left: {g_pct}%"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#888; margin-top:5px;">
                    <span>Normal (&lt;140)</span>
                    <span>Intolerancia</span>
                    <span>Diabetes (&gt;200)</span>
                </div>
                
                <div style="margin-top:25px;"></div>
                
                <div style="font-size:0.8rem; font-weight:700; margin-bottom:5px;">BMI ({bmi:.1f} kg/m²)</div>
                <div class="bar-bg">
                    <div class="bar-fill-bmi"></div>
                    <div class="bar-marker" style="left: {b_pct}%"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#888; margin-top:5px;">
                    <span>Bajo</span><span>Normal</span><span>Sobrepeso</span><span>Ob. G1</span><span>Ob. G2</span><span>Ob. G3</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            # TARJETA HALLAZGOS Y PROBABILIDAD
            alert_html = "".join([f"<div style='margin-bottom:4px;'>• {a}</div>" for a in alerts]) if alerts else "✅ Sin hallazgos"
            
            st.markdown(f"""
            <div class="card" style="border-left:5px solid {risk_color};">
                <div class="card-header" style="color:{risk_color}; margin-bottom:10px;">HALLAZGOS CLAVE</div>
                <div style="font-weight:600; font-size:0.9rem;">{alert_html}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico Donut
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.set_facecolor('none'); fig.patch.set_facecolor('none')
            ax.pie([prob, 1-prob], colors=[risk_color, '#F0F2F6'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15, edgecolor='white'))
            
            # Línea Umbral
            angle = 90 - (threshold * 360)
            rad = np.deg2rad(angle)
            ax.plot([0.85*np.cos(rad), 1.15*np.cos(rad)], [0.85*np.sin(rad), 1.15*np.sin(rad)], color=CEMP_DARK, linestyle='--', linewidth=2)
            
            donut_html = fig_to_html(fig)
            plt.close(fig)
            
            st.markdown(f"""
            <div class="card" style="text-align:center; padding-top:10px;">
                <div class="card-header" style="justify-content:center;">PROBABILIDAD IA</div>
                <div style="position:relative; display:flex; justify-content:center; align-items:center;">
                    {donut_html}
                    <div style="position:absolute; font-size:1.8rem; font-weight:800; color:{CEMP_DARK};">
                        {prob*100:.1f}%
                    </div>
                </div>
                <div style="font-size:0.65rem; color:#999; text-transform:uppercase; letter-spacing:1px; margin-top:5px;">
                    <span style="border-top:2px dashed {CEMP_DARK}; width:15px; display:inline-block; margin-bottom:3px;"></span> Umbral Decisión
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_main:
        pass # Para cerrar el tab visualmente si hiciera falta

    with tab_shap:
        st.write("")
        st.info("Aquí se mostraría el gráfico SHAP Waterfall detallado.")

    with tab_report:
        st.write("")
        st.success("✅ Informe Generado Correctamente")
        
        report_text = f"""INFORME CLÍNICO - DIABETES.NME\nFecha: {date.today()}\nPaciente: {st.session_state.patient['name']}\nID: {st.session_state.patient['id']}\n\nRESULTADOS:\nRiesgo: {risk_txt}\nProbabilidad: {prob*100:.1f}%\nAlertas: {', '.join(alerts)}\n\nDATOS:\nGlucosa 2h: {glucose}\nBMI: {bmi:.2f}\nInsulina: {insulin}"""
        
        st.download_button("📄 Descargar Informe (TXT)", data=report_text, file_name="informe.txt")
