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

# --- 2. PALETA DE COLORES (IDENTITY) ---
CEMP_PINK = "#E97F87"
CEMP_DARK = "#2C3E50"      # Azul noche
CEMP_BG_LIGHT = "#F8F9FA"  # Gris muy suave para fondos
GOOD_TEAL = "#4DB6AC"
TEXT_GRAY = "#6C757D"

# Gradientes
BMI_GRADIENT = "linear-gradient(90deg, #81D4FA 0%, #4DB6AC 25%, #FFF176 40%, #FFB74D 55%, #E97F87 70%, #880E4F 100%)"
GLUCOSE_GRADIENT = "linear-gradient(90deg, #4DB6AC 0%, #4DB6AC 28%, #FFF176 32%, #FFB74D 48%, #E97F87 52%, #880E4F 100%)"
RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

# --- 3. CSS PROFESIONAL ---
st.markdown(f"""
    <style>
    /* IMPORTAR FUENTE (Opcional, usa la del sistema por defecto para velocidad) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* OCULTAR ELEMENTOS NATIVOS MOLESTOS */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* CONTENEDOR PRINCIPAL */
    .block-container {{
        max-width: 1200px; 
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* --- LOGO EN SIDEBAR --- */
    .cemp-logo {{ 
        font-weight: 900; 
        font-size: 1.8rem; 
        color: {CEMP_DARK}; 
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }}
    .cemp-logo span {{ color: {CEMP_PINK}; }}
    .cemp-subtitle {{
        font-size: 0.7rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 20px;
        font-weight: 600;
    }}

    /* --- ESTILO DE TARJETAS (CARDS) --- */
    .card {{
        background-color: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.03);
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }}
    
    /* TARJETA DE ADMISIÓN (LOGIN STYLE) */
    .admission-card {{
        background-color: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border-top: 6px solid {CEMP_PINK};
        text-align: center;
    }}
    
    .admission-title {{
        font-size: 1.5rem;
        color: {CEMP_DARK};
        font-weight: 800;
        margin-bottom: 10px;
    }}
    
    .admission-desc {{
        color: {TEXT_GRAY};
        font-size: 0.9rem;
        margin-bottom: 30px;
        line-height: 1.5;
    }}

    /* --- BOTONES --- */
    /* Botón Primario (Coral) */
    div.stButton > button:first-child {{
        background-color: {CEMP_PINK};
        color: white;
        font-size: 1rem;
        font-weight: 700;
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 10px rgba(233, 127, 135, 0.3);
        transition: all 0.3s;
        width: 100%;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #D66E76;
        box-shadow: 0 6px 15px rgba(233, 127, 135, 0.4);
        transform: translateY(-2px);
        color: white;
    }}

    /* --- SIDEBAR STYLING --- */
    [data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #F0F2F6;
    }}
    
    .sidebar-section {{
        font-size: 0.75rem;
        font-weight: 700;
        color: #999;
        text-transform: uppercase;
        margin-top: 20px;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }}

    /* INPUTS NUMÉRICOS BONITOS */
    [data-testid="stNumberInput"] input {{
        text-align: center;
        font-weight: 600;
        color: {CEMP_DARK};
        border-radius: 8px;
    }}

    /* GRÁFICOS Y BARRAS */
    .bar-bg {{ background: #F0F2F5; height: 12px; border-radius: 6px; width: 100%; overflow: hidden; }}
    .bar-fill {{ height: 100%; width: 100%; background: {RISK_GRADIENT}; border-radius: 6px; }}
    .bar-fill-bmi {{ height: 100%; width: 100%; background: {BMI_GRADIENT}; border-radius: 6px; }}
    .bar-fill-glucose {{ height: 100%; width: 100%; background: {GLUCOSE_GRADIENT}; border-radius: 6px; }}
    
    .bar-marker {{ 
        position: absolute; top: -6px; width: 4px; height: 24px; 
        background: {CEMP_DARK}; border: 2px solid white; border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2); z-index: 10;
    }}
    
    /* UTILS */
    .highlight-box {{
        background: {NOTE_GRAY_BG};
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #EEE;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        font-size: 0.85rem;
        color: {NOTE_GRAY_TEXT};
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
    return f"""<span style="cursor:help; color:#999; font-size:0.8rem; margin-left:5px;" title="{description}">ⓘ</span>"""

def generate_report(data, prob, risk, alerts):
    # Generador simple de reporte texto
    return f"""INFORME CLÍNICO - DIABETES.NME
Fecha: {data['date']} | Paciente: {data['name']} (ID: {data['id']})
------------------------------------------------
RIESGO DETECTADO: {risk} (Probabilidad IA: {prob*100:.1f}%)
HALLAZGOS: {', '.join(alerts) if alerts else 'Ninguno'}
------------------------------------------------
DATOS: Glucosa 2h: {data['glucose']}, BMI: {data['bmi']:.2f}, Edad: {data['age']}
Este informe es una ayuda a la decisión clínica."""

# --- 5. LÓGICA DE ESTADO ---
if 'model' not in st.session_state:
    # MOCK MODEL
    class MockModel:
        def predict_proba(self, X):
            score = (X[0]*0.5) + (X[1]*0.4) + (X[3]*0.1) 
            prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

if 'page' not in st.session_state:
    st.session_state.page = 'admission' # 'admission' or 'dashboard'

if 'patient' not in st.session_state:
    st.session_state.patient = {'id': '', 'name': '', 'date': date.today()}

# --- 6. SIDEBAR (CONFIGURACIÓN CLÍNICA) ---
with st.sidebar:
    st.markdown(f'''
        <div class="cemp-logo">DIABETES<span>.NME</span></div>
        <div class="cemp-subtitle">Clinical Decision Support</div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">1. Bioquímica</div>', unsafe_allow_html=True)
    
    # Inputs compactos
    c1, c2 = st.columns([2, 1])
    with c1: st.markdown("**Glucosa 2h** (mg/dL)" + get_help_icon("Prueba de tolerancia"))
    with c2: glucose = st.number_input("", 50, 350, 120, key="glu_in", label_visibility="collapsed")
    
    c1, c2 = st.columns([2, 1])
    with c1: st.markdown("**Insulina** (µU/ml)")
    with c2: insulin = st.number_input("", 0, 900, 100, key="ins_in", label_visibility="collapsed")
    
    # Calculado
    ri = glucose * insulin
    st.markdown(f"""
    <div style="background:#F0F4F8; padding:8px 12px; border-radius:6px; margin-top:5px; margin-bottom:15px; display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:0.75rem; font-weight:600; color:#666;">ÍNDICE RI</span>
        <span style="font-size:0.9rem; font-weight:800; color:{CEMP_DARK};">{ri:,.0f}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">2. Antropometría</div>', unsafe_allow_html=True)
    
    weight = st.slider("Peso (kg)", 30.0, 200.0, 70.0, step=0.1)
    height = st.slider("Altura (m)", 1.00, 2.20, 1.70, step=0.01)
    
    bmi = weight / (height**2)
    st.markdown(f"""
    <div style="background:#F0F4F8; padding:8px 12px; border-radius:6px; margin-top:-5px; display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:0.75rem; font-weight:600; color:#666;">BMI (kg/m²)</span>
        <span style="font-size:0.9rem; font-weight:800; color:{CEMP_DARK};">{bmi:.2f}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">3. Historia</div>', unsafe_allow_html=True)
    age = st.slider("Edad", 18, 90, 45)
    pregnancies = st.slider("Embarazos", 0, 15, 1)
    dpf = st.slider("Carga Genética (DPF)", 0.0, 2.5, 0.5, step=0.01)

# --- 7. PÁGINA DE ADMISIÓN (HOME) ---
if st.session_state.page == 'admission':
    
    # Espaciado vertical para centrar
    st.write("") 
    st.write("")
    
    # Columnas para centrar la tarjeta de admisión
    col_l, col_c, col_r = st.columns([1, 2, 1])
    
    with col_c:
        # INICIO TARJETA
        st.markdown(f"""
        <div class="admission-card">
            <div style="font-size:3rem; margin-bottom:10px;">🩺</div>
            <div class="admission-title">Módulo de Admisión de Pacientes</div>
            <div class="admission-desc">
                Bienvenido al Sistema de Soporte a la Decisión Clínica (CDSS). 
                Por favor, registre los datos administrativos del paciente a continuación y verifique 
                las variables clínicas en el panel lateral antes de procesar el análisis de riesgo.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Separador
        
        # FORMULARIO DE DATOS
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.patient['id'] = st.text_input("ID / Historia Clínica", placeholder="Ej: 8842-X")
            with c2:
                st.session_state.patient['date'] = st.date_input("Fecha de Consulta", value=date.today())
            
            st.session_state.patient['name'] = st.text_input("Nombre Completo (Opcional)", placeholder="Nombre del paciente")

        st.write("") 
        st.write("") 
        
        # BOTÓN DE ACCIÓN PRINCIPAL
        if st.button("ANALIZAR RIESGO CLÍNICO  ➔"):
            if not st.session_state.patient['id']:
                st.warning("⚠️ Por favor, introduzca al menos un ID de paciente.")
            else:
                with st.spinner("Procesando biomarcadores y ejecutando modelo predictivo..."):
                    time.sleep(1.5) # Simular carga para efecto "wow"
                    st.session_state.page = 'dashboard'
                    st.rerun()

# --- 8. PÁGINA DE DASHBOARD (RESULTADOS) ---
elif st.session_state.page == 'dashboard':
    
    # Header minimalista con botón de "Atrás"
    c_head1, c_head2 = st.columns([3, 1])
    with c_head1:
        st.markdown(f"<h1 style='margin:0; color:{CEMP_DARK};'>Resultados del Análisis</h1>", unsafe_allow_html=True)
        st.caption(f"Paciente: {st.session_state.patient['name']} ({st.session_state.patient['id']}) | Fecha: {st.session_state.patient['date'].strftime('%d/%m/%Y')}")
    with c_head2:
        if st.button("🔄 Nueva Consulta"):
            st.session_state.page = 'admission'
            st.session_state.patient = {'id': '', 'name': '', 'date': date.today()}
            st.rerun()
            
    st.markdown("---")

    # --- PESTAÑAS DE CONTENIDO ---
    tab1, tab2, tab3 = st.tabs(["Panel Clínico", "Factores (SHAP)", "Informe"])

    with tab1:
        st.write("")
        
        # 1. CONFIGURACIÓN DE UMBRAL (EXPANDIBLE MEJORADO)
        with st.expander("⚙️ Calibración del Modelo (Umbral de Decisión)", expanded=False):
            ce1, ce2 = st.columns([1, 2])
            with ce1:
                st.caption("Ajuste manual de sensibilidad.")
                threshold = st.slider("Umbral", 0.0, 1.0, 0.27, 0.01, label_visibility="collapsed")
                st.markdown(f"""
                <div class="highlight-box">
                    <div style="font-size:1.2rem;">💡</div>
                    <div><strong>Criterio Técnico:</strong> Se recomienda <strong>0.27</strong> (F2-Score) para priorizar la detección de casos positivos y minimizar falsos negativos en cribado.</div>
                </div>""", unsafe_allow_html=True)
            
            with ce2:
                # GRÁFICA DE DENSIDADES
                x = np.linspace(-0.15, 1.25, 500)
                y_sanos = 1.9 * np.exp(-((x - 0.1)**2) / (2 * 0.11**2)) + 0.5 * np.exp(-((x - 0.55)**2) / (2 * 0.15**2))
                y_enfermos = 0.35 * np.exp(-((x - 0.28)**2) / (2 * 0.1**2)) + 1.4 * np.exp(-((x - 0.68)**2) / (2 * 0.16**2))
                
                fig_cal, ax_cal = plt.subplots(figsize=(6, 2))
                fig_cal.patch.set_facecolor('none')
                ax_cal.set_facecolor('none')
                ax_cal.fill_between(x, y_sanos, color="#BDC3C7", alpha=0.3, label="Clase 0: No Diabetes")
                ax_cal.plot(x, y_sanos, color="gray", lw=1, alpha=0.6)
                ax_cal.fill_between(x, y_enfermos, color=CEMP_PINK, alpha=0.3, label="Clase 1: Diabetes")
                ax_cal.plot(x, y_enfermos, color=CEMP_PINK, lw=1, alpha=0.6)
                ax_cal.axvline(0.27, color=OPTIMAL_GREEN, linestyle="--", linewidth=1.5, label="Óptimo (0.27)")
                ax_cal.axvline(threshold, color=CEMP_DARK, linestyle="--", linewidth=2, label="Selección")
                ax_cal.set_yticks([]); ax_cal.set_xlim(-0.2, 1.25)
                ax_cal.spines['top'].set_visible(False); ax_cal.spines['right'].set_visible(False); ax_cal.spines['left'].set_visible(False)
                ax_cal.legend(loc='upper right', fontsize=6, frameon=False)
                st.markdown(f'<div style="display:flex; justify-content:center;">{fig_to_html(fig_cal)}</div>', unsafe_allow_html=True)
                plt.close(fig_cal)

        # 2. LÓGICA DE PREDICCIÓN
        input_data = [glucose, bmi, insulin, age, pregnancies, dpf]
        prob = st.session_state.model.predict_proba(input_data)[0][1]
        is_high = prob > threshold
        
        # ALERTAS INTELIGENTES
        alerts = []
        if glucose >= 200: alerts.append("Posible Diabetes")
        elif glucose >= 140: alerts.append("Posible Prediabetes")
        
        if bmi >= 40: alerts.append("Obesidad Mórbida (G3)")
        elif bmi >= 30: alerts.append(f"Obesidad (BMI {bmi:.1f})")
        elif bmi >= 25: alerts.append("Sobrepeso")
        
        if ri > 19769: alerts.append("Resistencia Insulina")

        risk_color = CEMP_PINK if is_high else GOOD_TEAL
        risk_label = "RIESGO ALTO" if is_high else "RIESGO BAJO"
        risk_icon = "🔴" if is_high else "🟢"
        
        # 3. TARJETAS PRINCIPALES
        col_main1, col_main2 = st.columns([1.8, 1], gap="medium")
        
        # IZQUIERDA: Contexto Clínico
        with col_main1:
            # Contexto Poblacional
            g_pos = min(100, max(0, (glucose - 50) / 3.0))
            b_pos = min(100, max(0, (bmi - 10) * 2.5))
            
            st.markdown(f"""
            <div class="card">
                <div class="card-header">CONTEXTO POBLACIONAL</div>
                
                <div style="margin-top:10px; margin-bottom:5px; font-weight:700; color:#555; font-size:0.8rem;">
                    GLUCOSA 2H (PTOG) <span style="font-weight:400; color:#888;">({glucose} mg/dL)</span>
                </div>
                <div class="bar-container">
                    <div class="bar-bg"><div class="bar-fill-glucose"></div></div>
                    <div class="bar-marker" style="left: {g_pos}%;"></div>
                    <div class="bar-txt" style="left: {g_pos}%;">{glucose}</div>
                </div>
                <div class="legend-container">
                    <span class="legend-label" style="left: 15%;">Normal (&lt;140)</span>
                    <span class="legend-label" style="left: 40%;">Intolerancia</span>
                    <span class="legend-label" style="left: 75%;">Diabetes (&gt;200)</span>
                </div>

                <div style="margin-top:40px;"></div>

                <div style="margin-bottom:5px; font-weight:700; color:#555; font-size:0.8rem;">
                    ÍNDICE DE MASA CORPORAL <span style="font-weight:400; color:#888;">({bmi:.1f})</span>
                </div>
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
            """, unsafe_allow_html=True)

        # DERECHA: Resultado Modelo
        with col_main2:
            # Alertas
            alert_html = ""
            if alerts:
                for a in alerts:
                    alert_html += f"<div>• {a}</div>"
            else:
                alert_html = "<div>✅ Sin hallazgos significativos</div>"

            st.markdown(f"""
            <div class="card" style="min-height: auto; border-left: 5px solid {risk_color};">
                <div class="card-header" style="color:{risk_color};">HALLAZGOS CLAVE</div>
                <div style="font-weight:600; color:{CEMP_DARK}; font-size:0.9rem;">
                    {alert_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilidad IA
            fig, ax = plt.subplots(figsize=(3, 3))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            ax.pie([prob, 1-prob], colors=[risk_color, '#F0F2F6'], startangle=90, counterclock=False, 
                   wedgeprops=dict(width=0.15, edgecolor='white'))
            
            # Linea Umbral
            th_angle = 90 - (threshold * 360)
            th_rad = np.deg2rad(th_angle)
            ax.plot([0.85*np.cos(th_rad), 1.15*np.cos(th_rad)], [0.85*np.sin(th_rad), 1.15*np.sin(th_rad)], 
                    color=CEMP_DARK, linestyle='--', linewidth=2)
            
            chart = fig_to_html(fig)
            plt.close(fig)

            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="card-header" style="justify-content:center;">PROBABILIDAD (IA)</div>
                <div style="position:relative; display:flex; justify-content:center; align-items:center;">
                    {chart}
                    <div style="position:absolute; font-size:1.8rem; font-weight:800; color:{CEMP_DARK};">
                        {prob*100:.1f}%
                    </div>
                </div>
                <div style="margin-top:10px; font-size:0.7rem; color:#999; font-weight:600; display:flex; justify-content:center; gap:5px;">
                    <span style="border-top:2px dashed {CEMP_DARK}; width:15px; display:inline-block; position:relative; top:-4px;"></span>
                    UMBRAL DE DECISIÓN
                </div>
                <div style="margin-top:15px; background:{risk_bg}; color:{risk_color}; padding:5px 10px; border-radius:20px; font-weight:800; font-size:0.8rem; display:inline-block;">
                    {risk_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.write("")
        st.info("Aquí iría el gráfico SHAP explicativo.")
    
    with tab3:
        st.write("")
        st.success("✅ Informe listo para descargar.")
        
        # Datos para reporte
        report_data = st.session_state.patient.copy()
        report_data.update({'glucose': glucose, 'bmi': bmi, 'age': age, 'proxy_index': ri, 'weight': weight, 'height': height, 'pregnancies': pregnancies, 'dpf': dpf, 'insulin': insulin})
        
        report_txt = generate_report(report_data, prob, risk_label, alerts)
        
        st.download_button(
            label="📄 Descargar Informe Clínico (TXT)",
            data=report_txt,
            file_name=f"Informe_{st.session_state.patient['id']}.txt",
            mime="text/plain"
        )
