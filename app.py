import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import datetime
import joblib
import os

# Intentamos importar SHAP de forma segura
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# =========================================================
# 1. CONFIGURACIÓN Y CLASES (GLOBAL)
# =========================================================
st.set_page_config(
    page_title="DIABETES.NME", 
    page_icon="🩺", 
    layout="wide"
)

class MockModel:
    def predict_proba(self, X):
        # Simulación simple
        if isinstance(X, pd.DataFrame):
            score = (X.iloc[0]['Glucose']*0.5) + (X.iloc[0]['BMI']*0.4) + (X.iloc[0]['Age']*0.1) 
        else:
            score = 50
        prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
        return [[1-prob, prob]]

# --- FUNCIÓN DE CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    model_path = "modelos/diabetes_rf_pipeline.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error cargando modelo: {e}")
            return MockModel()
    else:
        return MockModel()

# Inicialización segura
if 'model' not in st.session_state:
    st.session_state.model = load_model()

if 'predict_clicked' not in st.session_state:
    st.session_state.predict_clicked = False

# Variable de estado para guardar los datos SHAP entre pestañas
if 'shap_data' not in st.session_state:
    st.session_state.shap_data = None

# =========================================================
# 2. FUNCIONES AUXILIARES
# =========================================================

def fig_to_html(fig):
    """Convierte una figura de Matplotlib a string HTML base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%; object-fit:contain;">'

def fig_to_bytes(fig):
    """Convierte figura a bytes para st.image (Permite zoom con fondo BLANCO)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=False, facecolor='white', dpi=300)
    buf.seek(0)
    return buf

def get_help_icon(description):
    return f"""<span style="display:inline-block; width:16px; height:16px; line-height:16px; text-align:center; border-radius:50%; background:#E0E0E0; color:#777; font-size:0.7rem; font-weight:bold; cursor:help; margin-left:6px; position:relative; top:-1px;" title="{description}">?</span>"""

def create_html_report(patient_name, date_str, prob, risk_label, inputs_dict, shap_rows_html, recommendation):
    """Genera un informe HTML completo con logo correcto y tabla SHAP."""
    
    c_dark = "#2C3E50"
    c_pink = "#E97F87"
    c_teal = "#4DB6AC"
    c_gray = "#bdc3c7"
    c_bg_head = "#F8F9FA"
    
    risk_color = c_pink if "ALTO" in risk_label else c_teal
    
    # Construir tabla de datos clínicos
    inputs_rows = ""
    for k, v in inputs_dict.items():
        inputs_rows += f"<tr><td style='padding:8px; border-bottom:1px solid #eee; color:#666;'>{k}</td><td style='padding:8px; border-bottom:1px solid #eee; font-weight:bold; color:{c_dark}; text-align:right;'>{v}</td></tr>"

    # Construir sección SHAP (Explicabilidad)
    shap_section_html = ""
    if shap_rows_html:
        shap_section_html = f"""
        <div class="section">
            <div class="section-title">Análisis Individual: Factores Determinantes (SHAP)</div>
            <p style="font-size:12px; color:#666; margin-bottom:15px;">Desglose de las variables que más han influido (positiva o negativamente) en el cálculo del riesgo para este paciente concreto.</p>
            <table style="width:100%; font-size:12px;">
                <thead>
                    <tr style="background:#eee; color:#777;">
                        <th style="padding:8px; text-align:left;">Variable Clínica</th>
                        <th style="padding:8px; text-align:center;">Valor Paciente</th>
                        <th style="padding:8px; text-align:right;">Contribución al Riesgo</th>
                    </tr>
                </thead>
                <tbody>
                    {shap_rows_html}
                </tbody>
            </table>
        </div>
        """

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Helvetica', sans-serif; color: #333; padding: 40px; max-width: 800px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid {c_pink}; padding-bottom: 20px; }}
            /* ESTILOS DEL LOGO EXACTOS */
            .logo {{ font-size: 26px; font-weight: 800; color: {c_dark}; letter-spacing: -1px; }}
            .logo span.pink {{ color: {c_pink}; }}
            .logo span.gray {{ color: {c_gray}; }}
            
            .meta {{ font-size: 11px; color: #999; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }}
            .patient-info {{ background: {c_bg_head}; padding: 15px; border-radius: 8px; margin-bottom: 30px; display: flex; justify-content: space-between; }}
            .section {{ margin-bottom: 30px; }}
            .section-title {{ color: {c_dark}; font-size: 14px; font-weight: 800; text-transform: uppercase; margin-bottom: 12px; border-left: 4px solid {c_pink}; padding-left: 10px; }}
            .risk-box {{ padding: 25px; background: {risk_color}15; border-radius: 12px; border: 2px solid {risk_color}; text-align: center; margin-bottom: 30px; }}
            .risk-val {{ font-size: 42px; font-weight: 900; color: {risk_color}; line-height: 1; }}
            .risk-txt {{ font-size: 16px; font-weight: 800; color: {c_dark}; text-transform: uppercase; margin-top: 10px; }}
            .rec-box {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 8px; font-size: 13px; line-height: 1.5; border-left: 4px solid {risk_color}; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">D<span class="pink">IA</span>BETES<span class="gray">.</span><span class="pink">NME</span></div>
            <div class="meta">Clinical Decision Support System Report</div>
        </div>
        
        <div class="patient-info">
            <div><span style="color:#999; font-size:11px;">PACIENTE:</span><br><strong>{patient_name}</strong></div>
            <div style="text-align:right;"><span style="color:#999; font-size:11px;">FECHA:</span><br><strong>{date_str}</strong></div>
        </div>

        <div class="risk-box">
            <div class="risk-val">{prob*100:.1f}%</div>
            <div class="risk-txt">{risk_label}</div>
            <div style="font-size:12px; color:#777; margin-top:8px; font-weight:500;">Probabilidad estimada de Diabetes Tipo 2</div>
        </div>

        {shap_section_html}

        <div class="section">
            <div class="section-title">Protocolo de Acción Recomendado</div>
            <div class="rec-box">
                <strong>Intervención Sugerida:</strong><br>
                {recommendation}
            </div>
        </div>

        <div class="section">
            <div class="section-title">Datos Clínicos Registrados</div>
            <table>
                {inputs_rows}
            </table>
        </div>

        <div style="font-size:10px; color:#aaa; text-align:center; margin-top:50px; border-top:1px solid #eee; padding-top:15px; line-height:1.4;">
            Documento generado automáticamente por el prototipo DIABETES.NME.<br>
            Herramienta de soporte a la decisión clínica desarrollada con fines académicos. <strong>No sustituye el criterio médico profesional.</strong><br>
            <strong style="color:{c_dark};">TFM Desarrollado por: Nerea Moreno Escamilla</strong> | CEMP 2025
        </div>
    </body>
    </html>
    """
    return html

# =========================================================
# 3. NAVEGACIÓN
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "landing"

def ir_a_simulacion():
    st.session_state.page = "simulacion"

def volver_inicio():
    st.session_state.page = "landing"

# =========================================================
# 4. PÁGINA: PORTADA
# =========================================================
if st.session_state.page == "landing":
    st.markdown("""
    <style>
        .stApp { background-color: #f0f2f6; }
        #MainMenu, footer, header { visibility: hidden; }
        .block-container {
            background-color: white; padding: 3rem !important;
            border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            max-width: 800px !important; margin-top: 2rem;
            margin-left: auto !important; margin-right: auto !important;
        }
        h1 {
            text-align: center; font-family: 'Helvetica', sans-serif !important;
            font-weight: 800 !important; font-size: 3.5rem !important;
            color: #2c3e50 !important; 
            margin-bottom: 0 !important;
            line-height: 1.2 !important; letter-spacing: -1px; cursor: default;
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
            text-align: justify; color: #666; line-height: 1.6;
            font-size: 0.95rem; margin-bottom: 30px; padding: 0 20px;
        }
        .warning-box {
            background-color: #f9fafb; border-left: 4px solid #ef7d86;
            padding: 20px; border-radius: 4px; font-size: 0.85rem;
            color: #555; margin-bottom: 30px; text-align: center;
        }
        .warning-box p { margin: 0; line-height: 1.5; }
        div.stButton > button {
            position: relative;
            background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%);
            color: white; border: none; padding: 14px 80px;
            border-radius: 50px; font-weight: bold; font-size: 14px;
            text-transform: uppercase; letter-spacing: 1px; white-space: nowrap;
            box-shadow: 0 4px 15px rgba(239,112,122,0.3); cursor: pointer;
            overflow: visible;
        }
        div.stButton > button > span {
            position: absolute; left: 50%; top: 50%;
            transform: translate(-50%, -50%) translateX(4px);
            display: inline-block; pointer-events: none;
        }
        div.stButton > button::after {
            content: "➔"; position: absolute; right: 28px; top: 50%;
            transform: translateY(-50%); font-size: 16px; pointer-events: none;
        }
        div.stButton > button:hover {
            transform: translateY(-2px); box-shadow: 0 6px 20px rgba(239,112,122,0.5); color: white;
        }
        @media (max-width: 600px) {
            div.stButton > button { padding: 12px 40px; font-size: 13px; }
            div.stButton > button::after { right: 18px; }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
<div class="badge-container">
<span class="badge">TFM • MÁSTER EN APLICACIONES DE LA INTELIGENCIA ARTIFICIAL EN LA SANIDAD</span>
</div>
<div class="institution">CENTRO EUROPEO DE MÁSTERES Y POSGRADOS</div>
<h1>D<span class="landing-pink">IA</span>BETES<span class="landing-gray">.</span><span class="landing-pink">NME</span></h1>
<div class="subtitle">Prototipo de CDSS para el diagnóstico temprano de diabetes</div>
<p class="description">Este proyecto explora el potencial de integrar modelos predictivos avanzados en el flujo de trabajo clínico, visualizando un futuro donde la IA actúa como un potente aliado en la detección temprana y prevención de la diabetes tipo 2.</p>
<div class="warning-box">
    <p><strong>Aplicación desarrollada con fines exclusivamente educativos como parte de un Trabajo de Fin de Máster.</strong></p>
    <p style="margin-top:10px;">⚠️ Esta herramienta NO es un dispositivo médico certificado. Los resultados son una simulación académica y NO deben utilizarse para el diagnóstico real, tratamiento o toma de decisiones clínicas.</p>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("INICIAR          "):
            ir_a_simulacion()
            st.rerun()

# =========================================================
# 5. PÁGINA: SIMULACIÓN
# =========================================================
elif st.session_state.page == "simulacion":

    CEMP_PINK = "#E97F87"
    CEMP_DARK = "#2C3E50" 
    GOOD_TEAL = "#4DB6AC"
    SLIDER_GRAY = "#BDC3C7"
    OPTIMAL_GREEN = "#8BC34A"
    NOTE_GRAY_BG = "#F8F9FA"  
    NOTE_GRAY_TEXT = "#6C757D" 
    URGENT_RED = "#C0392B"
      
    BMI_GRADIENT = "linear-gradient(90deg, #81D4FA 0%, #4DB6AC 25%, #FFF176 40%, #FFB74D 55%, #E97F87 70%, #880E4F 100%)"
    GLUCOSE_GRADIENT = "linear-gradient(90deg, #4DB6AC 0%, #4DB6AC 28%, #FFF176 32%, #FFB74D 48%, #E97F87 52%, #880E4F 100%)"
    RISK_GRADIENT = f"linear-gradient(90deg, {GOOD_TEAL} 0%, #FFD54F 50%, {CEMP_PINK} 100%)"

    st.markdown(f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .block-container {{
            max-width: 1250px; padding-top: 2rem; padding-bottom: 2rem; margin: 0 auto;
        }}
        
        [data-testid="stMarkdownContainer"] h1 a, 
        [data-testid="stMarkdownContainer"] h2 a, 
        [data-testid="stMarkdownContainer"] h3 a, 
        [data-testid="stMarkdownContainer"] h4 a {{
            display: none !important;
        }}
        
        .cemp-logo {{ 
            font-family: 'Helvetica', sans-serif; font-weight: 800; font-size: 1.8rem; color: {CEMP_DARK}; margin: 0; 
        }}
        .cemp-logo span {{ color: {CEMP_PINK}; }}
        .stSlider {{ padding-top: 0px !important; padding-bottom: 10px !important; }}
        div[data-testid="stExpander"] details > summary {{
            background-color: rgba(233, 127, 135, 0.1) !important;
            border: 1px solid rgba(233, 127, 135, 0.2) !important;
            border-radius: 8px !important;
            color: {CEMP_DARK} !important; font-weight: 700 !important; transition: background-color 0.3s;
        }}
        div[data-testid="stExpander"] details > summary:hover {{
            background-color: rgba(233, 127, 135, 0.2) !important; color: {CEMP_DARK} !important;
        }}
        div[data-testid="stExpander"] details > summary svg {{
            fill: {CEMP_DARK} !important; color: {CEMP_DARK} !important;
        }}
        div[data-testid="stExpander"] details[open] > div {{
            border-left: 1px solid rgba(233, 127, 135, 0.2);
            border-right: 1px solid rgba(233, 127, 135, 0.2);
            border-bottom: 1px solid rgba(233, 127, 135, 0.2);
            border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;
        }}
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
            padding: 0px 5px; font-size: 0.9rem; text-align: center; color: {CEMP_DARK};
            font-weight: 800; border-radius: 8px; background-color: white; border: 1px solid #ddd;
        }}
        [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {{ vertical-align: middle; }}
        .calc-box {{
            background-color: #F8F9FA; border-radius: 8px; padding: 12px 15px;
            border: 1px solid #EEE; margin-top: 5px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }}
        .calc-label {{
            font-size: 0.75rem; color: #888; font-weight: 600; text-transform: uppercase;
        }}
        .calc-value {{
            font-size: 1rem; color: {CEMP_DARK}; font-weight: 800;
        }}
        .card {{
            background-color: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.04);
            margin-bottom: 15px; display: flex; flex-direction: column; justify-content: center; min-height: 300px; 
        }}
        .card-auto {{ min-height: auto !important; height: 100%; }}
        .card-header {{
            color: #999; font-size: 0.75rem; font-weight: bold; letter-spacing: 1px;
            text-transform: uppercase; margin-bottom: 15px; display: flex; align-items: center;
        }}
        .bar-container {{ position: relative; width: 100%; margin-top: 20px; margin-bottom: 30px; }}
        .bar-bg {{ background: #F0F2F5; height: 12px; border-radius: 6px; width: 100%; overflow: hidden; }}
        .bar-fill {{ height: 100%; width: 100%; border-radius: 6px; opacity: 1; }}
        .bar-marker {{ 
            position: absolute; top: -6px; width: 4px; height: 24px; 
            background: {CEMP_DARK}; border: 1px solid white; border-radius: 2px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10; transition: left 0.3s ease;
        }}
        .bar-txt {{ 
            position: absolute; top: -30px; transform: translateX(-50%); 
            font-size: 0.85rem; font-weight: bold; color: {CEMP_DARK}; 
            background: white; padding: 2px 8px; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .legend-container {{ position: relative; width: 100%; height: 20px; margin-top: 8px; }}
        .legend-label {{
            position: absolute; transform: translateX(-50%); font-size: 0.7rem; 
            color: #888; font-weight: 600; text-align: center; white-space: nowrap;
        }}
        
        /* === ESTILOS PARA TARJETAS CON ZOOM === */
        .card-header-box {{
            background-color: white;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
            padding: 20px 25px 10px 25px;
            border-left: 1px solid #eee;
            border-right: 1px solid #eee;
            border-top: 1px solid #eee;
            text-align: center;
            margin-bottom: -5px; 
        }}
        .card-title-text {{
            color: #2C3E50;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-weight: 800;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 0;
        }}
        .card-footer-box {{
            background-color: rgba(233, 127, 135, 0.15); 
            padding: 20px 25px;
            color: #555;
            font-size: 0.9rem;
            line-height: 1.5;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
            border-left: 1px solid #eee;
            border-right: 1px solid #eee;
            border-bottom: 1px solid #eee;
            margin-top: -5px; 
            text-align: justify; /* <--- AQUI SE JUSTIFICA EL TEXTO DE INTERPRETACIÓN */
        }}
        
        /* ESTILOS PARA LA TABLA DE MÉTRICAS */
        .metrics-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 0.85rem; 
            margin-top: 15px;
        }}
        .metrics-table th {{
            text-align: center;
            padding: 10px 8px; 
            font-weight: 700;
            border-bottom: 2px solid #eee;
        }}
        .metrics-table td {{
            padding: 8px 8px; 
            color: {CEMP_DARK};
            border-bottom: 1px solid #f0f0f0;
            text-align: center;
        }}
        .metric-name-col {{
            text-align: left !important;
            font-weight: 600;
            color: #555;
        }}
        
        /* Badges para cabeceras */
        .badge-standard {{
            background-color: rgba(233, 127, 135, 0.1); 
            color: {CEMP_PINK};
            border: 1px solid rgba(233, 127, 135, 0.2);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
            white-space: nowrap;
        }}
        .badge-optimal {{
            background-color: rgba(77, 182, 172, 0.1); 
            color: {GOOD_TEAL};
            border: 1px solid {GOOD_TEAL}; 
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700; 
            display: inline-block;
            white-space: nowrap;
        }}

        /* Resaltado de celdas */
        .highlight-optimal {{
            color: {GOOD_TEAL};
            font-weight: 800;
            font-size: 0.95rem;
        }}
        .highlight-row {{
            background-color: #FAFAFA;
        }}
        
        .tech-card-title {{
            color: {CEMP_PINK};
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 15px;
            font-size: 1rem;
            text-align: justify;
        }}

        /* === ESTILOS PARA LA MATRIZ DE ACCIÓN === */
        .matrix-container {{
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-top: 20px;
        }}
        .matrix-row {{
            display: flex;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            border: 1px solid #eee;
            overflow: hidden;
            transition: all 0.3s ease;
            opacity: 0.6;
        }}
        .matrix-row-active {{
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border: 1px solid #d0d0d0;
            transform: scale(1.02);
            opacity: 1 !important;
            z-index: 10;
        }}
        .matrix-result {{
            flex: 1.2;
            padding: 20px;
            border-right: 1px solid #f0f0f0;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .matrix-result h4 {{
            margin: 0 0 8px 0;
            font-size: 1rem;
            font-weight: 800;
            color: {CEMP_DARK};
        }}
        .matrix-result p {{
            margin: 0;
            font-size: 0.85rem;
            color: #777;
            line-height: 1.4;
            text-align: justify; /* <--- AQUI SE JUSTIFICA EL TEXTO DE LA MATRIZ */
        }}
        .matrix-action {{
            flex: 2;
            padding: 20px 25px;
            display: flex;
            align-items: center;
            background-color: #fafafa;
        }}
        .matrix-action ul {{
            margin: 0;
            padding-left: 20px;
            font-size: 0.9rem;
            color: #555;
            line-height: 1.6;
        }}
        .matrix-priority {{
            flex: 0.6;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.9rem;
            text-align: center;
            letter-spacing: 0.5px;
        }}
        </style>
    """, unsafe_allow_html=True)

    def input_biomarker(label_text, min_val, max_val, default_val, key, help_text="", format_str=None):
        label_html = f"**{label_text}**"
        if help_text:
            label_html += get_help_icon(help_text)
        st.markdown(label_html, unsafe_allow_html=True)
        
        c1, c2 = st.columns([2.5, 1], gap="small")
        input_type = type(default_val)
        min_val = input_type(min_val)
        max_val = input_type(max_val)
        step = 0.1 if input_type == float else 1

        if format_str is None:
            format_str = "%.2f" if input_type == float else "%d"

        if key not in st.session_state:
            st.session_state[key] = default_val

        def update_from_slider():
            st.session_state[key] = st.session_state[f"{key}_slider"]
            st.session_state[f"{key}_input"] = st.session_state[f"{key}_slider"]
            st.session_state.predict_clicked = False 
            st.session_state.shap_data = None # Reset SHAP
        
        def update_from_input():
            val = st.session_state[f"{key}_input"]
            if val < min_val: val = min_val
            if val > max_val: val = max_val
            st.session_state[key] = val
            st.session_state[f"{key}_slider"] = val 
            st.session_state.predict_clicked = False 
            st.session_state.shap_data = None # Reset SHAP

        with c1:
            st.slider(
                label="", min_value=min_val, max_value=max_val, step=step,
                key=f"{key}_slider", value=st.session_state[key], on_change=update_from_slider, label_visibility="collapsed"
            )
        with c2:
            st.number_input(
                label="", min_value=min_val, max_value=max_val, step=step,
                key=f"{key}_input", value=st.session_state[key], on_change=update_from_input, label_visibility="collapsed",
                format=format_str 
            )
        return st.session_state[key]

    with st.sidebar:
        if st.button("⬅ Volver"):
            volver_inicio()
            st.rerun()

        st.markdown(f'<div class="cemp-logo">D<span>IA</span>BETES<span style="color:{SLIDER_GRAY}">.</span><span>NME</span></div>', unsafe_allow_html=True)
        st.caption("CLINICAL DECISION SUPPORT SYSTEM")
        st.write("")
        
        # --- AQUÍ AÑADO EL text-align: justify AL AVISO DE LA SIDEBAR ---
        st.markdown("""
        <div style="background-color: #FFF3E0; border-left: 4px solid #FFB74D; padding: 10px; border-radius: 5px; font-size: 0.8rem; color: #E65100; margin-bottom: 20px; text-align: justify;">
            ⚠️ <strong>Importante:</strong> Por favor, asegúrese de rellenar todos los campos clínicos con precisión.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Datos del paciente**")
        
        def reset_on_change():
            st.session_state.predict_clicked = False
            st.session_state.shap_data = None

        patient_name = st.text_input("ID Paciente", value="Paciente #8842-X", label_visibility="collapsed", on_change=reset_on_change)
        default_date = datetime.date.today()
        consult_date = st.date_input("Fecha Predicción", value=default_date, label_visibility="collapsed", on_change=reset_on_change)
        
        meses_es = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun", 
                    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
        date_str = f"{consult_date.day} {meses_es[consult_date.month]} {consult_date.year}"

        st.markdown("---")
        
        glucose = input_biomarker("Glucosa 2h (mg/dL)", 50, 350, 50, "gluc", "Concentración plasmática a las 2h de test de tolerancia oral.", format_str="%d")
        insulin = input_biomarker("Insulina (µU/ml)", 0, 900, 0, "ins", "Insulina a las 2h de ingesta.", format_str="%d")
        
        proxy_index = int(glucose * insulin)
        proxy_str = f"{proxy_index}" 

        st.markdown(f"""
        <div class="calc-box" style="border-left: 4px solid {CEMP_PINK};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="calc-label">Índice RI (Glucosa x Insulina)</span>
                <span class="calc-value">{proxy_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        blood_pressure = input_biomarker("Presión Arterial (mm Hg)", 0, 150, 0, "bp", "Presión arterial diastólica.", format_str="%d")

        st.markdown("---") 

        weight = input_biomarker("Peso (kg)", 30.0, 250.0, 30.0, "weight", "Peso corporal actual.")
        height = input_biomarker("Altura (m)", 1.00, 2.20, 1.00, "height", "Altura en metros.")
        
        if height > 0:
            bmi = weight / (height * height)
        else:
            bmi = 0
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

        c_age, c_preg = st.columns(2)
        age = input_biomarker("Edad (años)", 18, 90, 18, "age", format_str="%d")
        pregnancies = input_biomarker("Embarazos", 0, 20, 0, "preg", "Nº veces embarazada.", format_str="%d") 
        
        st.markdown("---") 

        dpf = input_biomarker("Antecedentes Familiares (DPF)", 0.0, 2.5, 0.0, "dpf", "Estimación de predisposición genética.")

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


    st.markdown(f"<h1 style='color:{CEMP_DARK}; margin-bottom: 10px; font-size: 2.2rem;'>Evaluación de Riesgo Diabético</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Panel General", "Explicabilidad", "Framework de Acción", "Ficha Técnica"])

    with tab1:
        st.write("")
        
        with st.expander("Ajuste de Sensibilidad Clínica"):
            c_calib_1, c_calib_2 = st.columns([1, 2], gap="large")
            with c_calib_1:
                st.caption("Selecciona manualmente el umbral de decisión.")
                threshold = st.slider("Umbral", 0.0, 1.0, 0.27, 0.01, label_visibility="collapsed")
                
                # --- NUEVA FUNCIÓN: MODAL (POP-UP) CON EL MISMO ESTILO QUE TAB 4 ---
                @st.dialog("Ficha Técnica Resumida")
                def ver_metricas_modal():
                    # Aquí re-inyectamos los estilos para asegurar que se vean IGUAL dentro del modal
                    # y usamos la misma estructura HTML que en Tab 4.
                    st.markdown(f"""
                    <style>
                        /* ESTILOS PARA LA TABLA DE MÉTRICAS (Replicados para el Modal) */
                        .metrics-table {{
                            width: 100%;
                            border-collapse: separate;
                            border-spacing: 0;
                            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                            font-size: 0.85rem; 
                            margin-top: 15px;
                        }}
                        .metrics-table th {{
                            text-align: center;
                            padding: 10px 8px; 
                            font-weight: 700;
                            border-bottom: 2px solid #eee;
                            background-color: white;
                        }}
                        .metrics-table td {{
                            padding: 8px 8px; 
                            color: {CEMP_DARK};
                            border-bottom: 1px solid #f0f0f0;
                            text-align: center;
                        }}
                        .metric-name-col {{
                            text-align: left !important;
                            font-weight: 600;
                            color: #555;
                        }}
                        
                        /* Badges */
                        .badge-standard {{
                            background-color: rgba(233, 127, 135, 0.1); 
                            color: {CEMP_PINK};
                            border: 1px solid rgba(233, 127, 135, 0.2);
                            padding: 4px 10px;
                            border-radius: 20px;
                            font-size: 0.75rem;
                            font-weight: 700;
                            display: inline-block;
                            white-space: nowrap;
                        }}
                        .badge-optimal {{
                            background-color: rgba(77, 182, 172, 0.1); 
                            color: {GOOD_TEAL};
                            border: 1px solid {GOOD_TEAL}; 
                            padding: 4px 10px;
                            border-radius: 20px;
                            font-size: 0.75rem;
                            font-weight: 700; 
                            display: inline-block;
                            white-space: nowrap;
                        }}
                        
                        /* Resaltado */
                        .highlight-optimal {{
                            color: {GOOD_TEAL};
                            font-weight: 800;
                            font-size: 0.95rem;
                        }}
                        .highlight-row {{
                            background-color: #FAFAFA;
                        }}
                        
                        .tech-card-title {{
                            color: {CEMP_PINK};
                            font-weight: 800;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;
                            margin-bottom: 15px;
                            font-size: 1rem;
                            text-align: justify;
                        }}
                    </style>
                    
                    <div>
                        <div class="tech-card-title">Métricas de Rendimiento (Test)</div>
                        <p style="font-size:0.9rem; color:#666; margin-bottom:15px; text-align: justify;">
                            Evaluación sobre conjunto de test independiente (10 repeticiones). Se prioriza la <strong>Sensibilidad (Recall)</strong> para minimizar falsos negativos.
                        </p>
                        <table class="metrics-table">
                            <thead>
                                <tr>
                                    <th></th>
                                    <th><span class="badge-standard">Umbral Estándar (0.5)</span></th>
                                    <th><span class="badge-optimal">Umbral Óptimo (0.27)</span></th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td class="metric-name-col">Accuracy</td>
                                    <td>0.738</td>
                                    <td>0.719</td>
                                </tr>
                                <tr>
                                    <td class="metric-name-col">Precision</td>
                                    <td>0.604</td>
                                    <td>0.560</td>
                                </tr>
                                <tr class="highlight-row">
                                    <td class="metric-name-col" style="color:{CEMP_DARK}; font-weight:800;">Recall (Sensibilidad)</td>
                                    <td>0.733</td>
                                    <td class="highlight-optimal">0.924</td>
                                </tr>
                                <tr class="highlight-row">
                                    <td class="metric-name-col" style="color:{CEMP_DARK}; font-weight:800;">F2-Score</td>
                                    <td>0.703</td>
                                    <td class="highlight-optimal">0.818</td>
                                </tr>
                                <tr>
                                    <td class="metric-name-col">AUC-ROC</td>
                                    <td>0.815</td>
                                    <td>0.815</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background-color:{NOTE_GRAY_BG}; margin-right: 15px; padding:15px; border-radius:8px; border:1px solid #E9ECEF; color:{NOTE_GRAY_TEXT}; font-size:0.85rem; display:flex; align-items:start; gap:10px; text-align: justify;">
                    <span style="font-size:1.1rem;">💡</span> 
                    <div>
                        <strong>Criterio Técnico:</strong> Se ha seleccionado <strong>0.27</strong> como umbral óptimo para minimizar los falsos negativos.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # BOTÓN QUE ABRE EL MODAL
                st.write("") 
                if st.button("📊 Ver Tabla de Métricas", type="secondary", use_container_width=True):
                    ver_metricas_modal()

            with c_calib_2:
                x = np.linspace(-0.15, 1.25, 500)
                y_sanos = 1.9 * np.exp(-((x - 0.1)**2) / (2 * 0.11**2)) + \
                          0.5 * np.exp(-((x - 0.55)**2) / (2 * 0.15**2))
                y_enfermos = 0.35 * np.exp(-((x - 0.28)**2) / (2 * 0.1**2)) + \
                             1.4 * np.exp(-((x - 0.68)**2) / (2 * 0.16**2))
                
                fig_calib, ax_calib = plt.subplots(figsize=(6, 2.5))
                # Fondo transparente para este gráfico pequeño que va sobre el fondo de la app
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
                
                st.image(fig_to_bytes(fig_calib), use_container_width=True)
                plt.close(fig_calib)

        # PREPARAR DATOS PARA EL MODELO REAL
        is_prediabetes = 1 if glucose >= 140 else 0
        
        # DataFrame con los nombres de columna EXACTOS
        input_data = pd.DataFrame([[
            pregnancies,
            glucose,
            blood_pressure,
            insulin,
            bmi,
            dpf,
            age,
            proxy_index, # Indice_resistencia
            bmi_sq,      # BMI_square
            is_prediabetes
        ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DPF', 'Age', 'Indice_resistencia', 'BMI_square', 'Is_prediabetes'])
        
        if 'model' in st.session_state and hasattr(st.session_state.model, 'predict_proba'):
            try:
                prob = st.session_state.model.predict_proba(input_data)[0][1]
            except:
                st.session_state.model = MockModel()
                prob = 0.5
        else:
            st.session_state.model = MockModel()
            prob = 0.5

        is_high = prob > threshold 
        
        distancia_al_corte = abs(prob - threshold)
        if distancia_al_corte > 0.15:
            conf_text, conf_color = "ALTA", GOOD_TEAL
            conf_desc = "Probabilidad claramente alejada del umbral."
        elif distancia_al_corte > 0.05:
            conf_text, conf_color = "MEDIA", "#F39C12"
            conf_desc = "Probabilidad relativamente cerca del umbral."
        else:
            conf_text, conf_color = "BAJA", CEMP_PINK
            conf_desc = "Zona de incertidumbre clínica."

        risk_color = CEMP_PINK if is_high else GOOD_TEAL
        risk_label = "ALTO RIESGO" if is_high else "BAJO RIESGO"
        risk_icon = "🔴" if is_high else "🟢"
        risk_bg = "#FFF5F5" if is_high else "#F0FDF4"
        risk_border = CEMP_PINK if is_high else GOOD_TEAL
        
        alerts = []
        if glucose >= 200: alerts.append("Posible Diabetes")
        elif glucose >= 140: alerts.append("Posible Prediabetes")
        if bmi >= 40: alerts.append("Obesidad Mórbida (G3)")
        elif bmi >= 35: alerts.append("Obesidad G2")
        elif bmi >= 30: alerts.append("Obesidad G1")
        elif bmi >= 25: alerts.append("Sobrepeso")
        elif bmi < 18.5 and bmi > 0: alerts.append("Bajo Peso")
        if proxy_index > 19769.5: alerts.append("Resistencia Insulina")
        if blood_pressure > 90: alerts.append("Hipertensión Diastólica")
        
        if not alerts: insight_txt, insight_bd, alert_icon = "Sin hallazgos significativos", GOOD_TEAL, "✅"
        else: insight_txt, insight_bd, alert_icon = " • ".join(alerts), CEMP_PINK, "⚠️"

        c_left, c_right = st.columns([1.8, 1], gap="medium") 
        
        with c_left:
            if st.session_state.predict_clicked:
                badges_html = f"""<div style="background:{risk_bg}; border:1px solid {risk_border}; color:{risk_border}; font-weight:bold; font-size:0.9rem; padding:8px 16px; border-radius:30px;">{risk_icon} {risk_label}</div><div style="background:#F8F9FA; border-radius:8px; padding: 4px 10px; border:1px solid #EEE; margin-top:5px;" title="{conf_desc}"><span style="font-size:0.7rem; color:#999; font-weight:600;">FIABILIDAD: </span><span style="font-size:0.75rem; color:{conf_color}; font-weight:800;">{conf_text}</span></div>"""
            else:
                badges_html = """<div style="color:#BDC3C7; font-size:0.8rem; font-weight:600; padding:10px; font-style:italic;">Análisis pendiente...</div>"""

            st.markdown(f"""
            <div class="card card-auto" style="flex-direction:row; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:20px; flex-grow:1;">
                    <div style="background:rgba(233, 127, 135, 0.1); width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2rem; color:{CEMP_DARK};">👤</div>
                    <div>
                        <span class="card-header" style="margin-bottom:5px;">EXPEDIENTE MÉDICO</span>
                        <h2 style="margin:0; color:{CEMP_DARK}; font-size:1.6rem; line-height:1.2;">{patient_name}</h2>
                        <div style="font-size:0.85rem; color:#666; margin-top:5px;">📅 Revisión: <b>{date_str}</b></div>
                    </div>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; gap:5px;">
                    {badges_html}
                </div>
            </div>""", unsafe_allow_html=True)

            g_pos = min(100, max(0, (glucose - 50) / 3.0)) 
            b_pos = min(100, max(0, (bmi - 10) * 2.5)) 
            
            st.markdown(f"""<div class="card">
                <span class="card-header">CONTEXTO POBLACIONAL</span>
                <div style="margin-top:15px;">
                    <div style="font-size:0.8rem; font-weight:bold; color:#666; margin-bottom:5px;">GLUCOSA 2H (TEST TOLERANCIA) <span style="font-weight:normal">({glucose} mg/dL)</span></div>
                    <div class="bar-container">
                        <div class="bar-bg"><div class="bar-fill" style="background: {GLUCOSE_GRADIENT};"></div></div>
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
                        <div class="bar-bg"><div class="bar-fill" style="background: {BMI_GRADIENT};"></div></div>
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
            st.markdown(f"""<div class="card card-auto" style="border-left:5px solid {insight_bd}; justify-content:center;">
                <span class="card-header" style="color:{insight_bd}; margin-bottom:10px;">HALLAZGOS CLAVE</span>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; color:{CEMP_DARK}; font-size:1.1rem; line-height:1.4;">{insight_txt}</h3>
                    <div style="font-size:1.8rem;">{alert_icon}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            
            # --- LÓGICA DE BOTONES Y CÁLCULO SHAP PARA INFORME ---
            if st.session_state.predict_clicked:
                
                # 1. Calcular SHAP para el informe (si está disponible)
                shap_html_rows = ""
                if SHAP_AVAILABLE and hasattr(st.session_state.model, 'named_steps'):
                    try:
                        pipeline = st.session_state.model
                        step1 = pipeline.named_steps['imputer'].transform(input_data)
                        step2 = pipeline.named_steps['scaler'].transform(step1)
                        model_step = pipeline.named_steps['model']
                        explainer = shap.TreeExplainer(model_step)
                        shap_values = explainer.shap_values(step2)
                        
                        # Obtener valores para la clase positiva (Diabetes)
                        if isinstance(shap_values, list): shap_val_instance = shap_values[1][0]
                        elif len(shap_values.shape) == 3: shap_val_instance = shap_values[0, :, 1]
                        else: shap_val_instance = shap_values[0]
                            
                        # Crear DataFrame y ordenar por impacto absoluto
                        df_shap = pd.DataFrame({
                            'Feature': input_data.columns,
                            'Impact': shap_val_instance,
                            'Value': input_data.iloc[0].values
                        })
                        df_shap['AbsImpact'] = df_shap['Impact'].abs()
                        # Top 5 factores más influyentes
                        df_top_shap = df_shap.sort_values(by='AbsImpact', ascending=False).head(5)
                        
                        # Generar filas HTML
                        for _, row in df_top_shap.iterrows():
                            impact_val = row['Impact']
                            # Color: Rojo si aumenta riesgo, Verde si disminuye
                            color = "#C0392B" if impact_val > 0 else "#27AE60"
                            sign = "+" if impact_val > 0 else ""
                            # Limpiar nombre de la variable
                            feat_name_clean = row['Feature'].replace('Indice_resistencia', 'Índice RI').replace('BMI_square', 'BMI² (No lineal)').replace('BloodPressure', 'Presión Arterial').replace('Pregnancies', 'Embarazos').replace('Age', 'Edad').replace('Glucose', 'Glucosa 2h').replace('Insulin', 'Insulina').replace('Is_prediabetes', 'Prediabetes Detectada')

                            shap_html_rows += f"""
                                <tr>
                                    <td style='padding:6px; border-bottom:1px solid #eee; font-weight:500;'>{feat_name_clean}</td>
                                    <td style='padding:6px; border-bottom:1px solid #eee; text-align:center; color:#666;'>{row['Value']:.2f}</td>
                                    <td style='padding:6px; border-bottom:1px solid #eee; text-align:right; font-weight:bold; color:{color};'>
                                        {sign}{impact_val:.3f} (log-odds)
                                    </td>
                                </tr>
                            """
                    except Exception as e:
                         print(f"Error SHAP Report: {e}")
                         shap_html_rows = "<tr><td colspan='3' style='text-align:center; color:#999; font-style:italic;'>Análisis detallado no disponible.</td></tr>"

                # 2. Determinar la recomendación activa
                active_rec = "Reevaluar en 3-6 meses."
                if glucose >= 200: active_rec = "Protocolo de confirmación diagnóstica urgente. Descartar cetoacidosis."
                elif is_high and distancia_al_corte > 0.05: active_rec = "Derivación a Endocrinología. Solicitar HbA1c y perfil lipídico. Valorar inicio de metformina."
                elif not is_high and distancia_al_corte <= 0.05: active_rec = "Repetir TTOG en 3-6 meses. Monitorización estrecha de glucemia basal."
                else: active_rec = "Seguimiento rutinario anual según guías locales. Mantener estilos de vida saludables."

                # 3. Generar el HTML del informe
                report_html = create_html_report(
                    patient_name, 
                    date_str, 
                    prob, 
                    risk_label, 
                    inputs_dict={
                        "Glucosa 2h": f"{glucose} mg/dL",
                        "Insulina 2h": f"{insulin} µU/ml",
                        "Índice RI (Glucosa x Insulina)": f"{proxy_index}", # Nombre corregido
                        "BMI": f"{bmi:.1f} kg/m²",
                        "Edad": f"{age} años",
                        "Presión Arterial": f"{blood_pressure} mm Hg",
                        "Embarazos": f"{pregnancies}",
                        "Carga Genética (DPF)": f"{dpf:.2f}"
                    },
                    shap_rows_html=shap_html_rows, # Pasamos las filas SHAP
                    recommendation=active_rec
                )
                
                # 4. Mostrar botón de descarga
                st.download_button(
                    label="📄 DESCARGAR INFORME CLÍNICO",
                    data=report_html,
                    file_name=f"CDSS_Diabetes_{patient_name.replace(' ', '_')}_{date_str.replace(' ', '_')}.html",
                    mime="text/html",
                    type="primary",
                    use_container_width=True
                )
            else:
                # Botón de cálculo inicial
                if st.button("CALCULAR RIESGO", use_container_width=True, type="primary"):
                    st.session_state.predict_clicked = True
                    st.rerun()

            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            if st.session_state.predict_clicked:
                ax.pie([prob, 1-prob], colors=[risk_color, '#F4F6F9'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15, edgecolor='none'))
                threshold_angle = 90 - (threshold * 360)
                theta_rad = np.deg2rad(threshold_angle)
                x1 = 0.85 * np.cos(theta_rad)
                y1 = 0.85 * np.sin(theta_rad)
                x2 = 1.15 * np.cos(theta_rad)
                y2 = 1.15 * np.sin(theta_rad)
                ax.plot([x1, x2], [y1, y2], color=CEMP_DARK, linestyle='--', linewidth=2)
                center_text = f"{prob*100:.1f}%"
            else:
                ax.pie([100], colors=['#EEEEEE'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15, edgecolor='none'))
                center_text = "---"

            chart_html = fig_to_html(fig) 
            plt.close(fig)
            
            prob_help = get_help_icon("Probabilidad calculada por el modelo de IA.")
            
            st.markdown(f"""<div class="card" style="text-align:center; justify-content: center;">
                <span class="card-header" style="justify-content:center; margin-bottom:15px;">PROBABILIDAD IA{prob_help}</span>
                <div style="position:relative; display:inline-block; margin: auto;">
                    {chart_html}
                    <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-size:2.5rem; font-weight:800; color:{CEMP_DARK}; letter-spacing:-1px;">
                        {center_text}
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 0.65rem; color: #999; display: flex; align-items: center; justify-content: center; gap: 5px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase;">
                    <span style="display: inline-block; width: 15px; border-top: 2px dashed {CEMP_DARK};"></span>
                    <span>Umbral de decisión</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.write("")
        
        # --- CABECERA CEREBRITO CON BORDE ROSA ---
        st.markdown(f"""
        <div style="background-color:#F8F9FA; padding:15px; border-radius:10px; border-left:5px solid {CEMP_PINK}; margin-bottom:20px;">
            <h4 style="margin:0; color:#2C3E50;">🧠 Inteligencia Artificial Explicable (XAI)</h4>
            <p style="margin:5px 0 0 0; color:#666; font-size:0.9rem;">
                Con el fin de ofrecer transparencia, a continuación, se detalla cómo son utilizadas las variables por el modelo predictivo.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c_exp1, c_exp2 = st.columns(2, gap="medium")
        
        # --- COLUMNA IZQUIERDA: POBLACIÓN GENERAL ---
        with c_exp1:
            st.markdown(f"""
            <div class="card-header-box">
                <div class="card-title-text">VISIÓN GLOBAL DEL MODELO</div>
            </div>
            """, unsafe_allow_html=True)
            
            if hasattr(st.session_state.model, 'named_steps'):
                try:
                    rf = st.session_state.model.named_steps['model']
                    importances = rf.feature_importances_
                    
                    feat_names_es = ['Embarazos', 'Glucosa', 'Presión Art.', 'Insulina', 'BMI', 'Ant. Familiares', 'Edad', 'Índice Resist.', 'BMI²', 'Prediabetes']
                    df_imp = pd.DataFrame({'Feature': feat_names_es, 'Importancia': importances})
                    df_imp = df_imp.sort_values(by='Importancia', ascending=True)
                    
                    fig_imp, ax_imp = plt.subplots(figsize=(6, 5))
                    fig_imp.patch.set_facecolor('white') 
                    ax_imp.set_facecolor('white')
                    
                    bars = ax_imp.barh(df_imp['Feature'], df_imp['Importancia'], color=CEMP_PINK, alpha=0.8)
                    ax_imp.spines['top'].set_visible(False)
                    ax_imp.spines['right'].set_visible(False)
                    ax_imp.spines['bottom'].set_visible(False)
                    ax_imp.spines['left'].set_visible(False)
                    ax_imp.tick_params(axis='y', colors=CEMP_DARK, labelsize=9)
                    ax_imp.tick_params(axis='x', colors='#999', labelsize=8)
                    
                    for bar in bars:
                        width = bar.get_width()
                        ax_imp.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                                    f'{width*100:.1f}%', ha='left', va='center', fontsize=8, color='#666')
                    
                    st.image(fig_to_bytes(fig_imp), use_container_width=True)
                    plt.close(fig_imp)

                except:
                    st.warning("No se pudo extraer la importancia global del modelo cargado.")
            else:
                st.warning("Modelo simulado: No hay datos reales de importancia global.")

            st.markdown(f"""
            <div class="card-footer-box">
                <span style="color: {CEMP_PINK}; font-weight: 800;">Interpretación del Modelo (General):</span><br>
                Este gráfico muestra qué <b>datos son más importantes</b> para la predicción del riesgo de padecer diabetes. Las <b>barras más largas</b> (como Glucosa o Índice RI) indican los <b>factores que más influyen</b> en el diagnóstico final para la población general.
            </div>
            """, unsafe_allow_html=True)

        # --- COLUMNA DERECHA: PACIENTE ESPECÍFICO ---
        with c_exp2:
            st.markdown(f"""
            <div class="card-header-box">
                <div class="card-title-text">ANÁLISIS INDIVIDUAL (SHAP)</div>
            </div>
            """, unsafe_allow_html=True)
            
            if SHAP_AVAILABLE and hasattr(st.session_state.model, 'named_steps') and st.session_state.predict_clicked:
                try:
                    pipeline = st.session_state.model
                    step1 = pipeline.named_steps['imputer'].transform(input_data)
                    step2 = pipeline.named_steps['scaler'].transform(step1)
                    
                    model_step = pipeline.named_steps['model']
                    explainer = shap.TreeExplainer(model_step)
                    shap_values = explainer.shap_values(step2)
                    
                    if isinstance(shap_values, list): shap_val_instance = shap_values[1][0]
                    elif len(shap_values.shape) == 3: shap_val_instance = shap_values[0, :, 1]
                    else: shap_val_instance = shap_values[0]
                    
                    if isinstance(explainer.expected_value, np.ndarray): base_value = explainer.expected_value[1]
                    else: base_value = explainer.expected_value

                    exp = shap.Explanation(
                        values=shap_val_instance,
                        base_values=base_value,
                        data=input_data.iloc[0].values, 
                        feature_names=input_data.columns
                    )
                    
                    fig_shap, ax_shap = plt.subplots(figsize=(6, 5))
                    fig_shap.patch.set_facecolor('white')
                    ax_shap.set_facecolor('white')

                    shap.plots.waterfall(exp, show=False, max_display=10)
                    plt.tight_layout()
                    
                    st.image(fig_to_bytes(fig_shap), use_container_width=True)
                    plt.close(fig_shap)

                except Exception as e:
                    st.error(f"Error generando SHAP: {e}")
            else:
                 st.markdown("""
                    <div style="display:flex; justify-content:center; align-items:center; height:300px; color:#aaa; font-style:italic;">
                        <div>Calcula el riesgo primero para ver el análisis individual.</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card-footer-box">
                <span style="color: {CEMP_PINK}; font-weight: 800;">Interpretación para {patient_name}:</span><br>
                El análisis parte de una <b>'Línea Base' (aprox. 50%)</b>. A este valor se le <b>suman (barras rojas)</b> o <b>restan (barras azules)</b> las contribuciones específicas de los datos del paciente. El <b>resultado final ({prob*100:.1f}%)</b> es la suma de estos factores.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.write("")
        # --- CABECERA DEL FRAMEWORK ---
        st.markdown(f"""
        <div style="background-color:#F8F9FA; padding:15px; border-radius:10px; border-left:5px solid {GOOD_TEAL}; margin-bottom:20px;">
            <h4 style="margin:0; color:#2C3E50;">👩🏻‍⚕️ Acción Clínica Recomendada </h4>
            <p style="margin:5px 0 0 0; color:#666; font-size:0.9rem;">
                Como marco de referencia para la toma de decisiones, el sistema destaca la intervención recomendada basándose en la predicción de la IA y los datos clínicos clave.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- LÓGICA PARA DETERMINAR EL ESCENARIO ACTIVO ---
        active_scenario = "bajo" # Default
        if st.session_state.predict_clicked:
            if glucose >= 200:
                active_scenario = "urgente"
            elif is_high and distancia_al_corte > 0.05:
                active_scenario = "alto"
            elif not is_high and distancia_al_corte > 0.05:
                active_scenario = "bajo"
            else:
                active_scenario = "incertidumbre"

        # --- DEFINICIÓN DE ESTILOS DE FILA ---
        style_urgente = "matrix-row-active" if active_scenario == "urgente" else ""
        style_alto = "matrix-row-active" if active_scenario == "alto" else ""
        style_incertidumbre = "matrix-row-active" if active_scenario == "incertidumbre" else ""
        style_bajo = "matrix-row-active" if active_scenario == "bajo" else ""
        
        # --- MATRIZ VISUAL (HTML SIN INDENTACIÓN) ---
        st.markdown(f"""
<div class="matrix-container">
<div class="matrix-row {style_urgente}" style="border-left: 5px solid {URGENT_RED};">
    <div class="matrix-result">
        <h4>Hiperglucemia Severa</h4>
        <p>Hallazgo crítico: Glucosa ≥ 200 mg/dL (Independiente del score IA).</p>
    </div>
    <div class="matrix-action">
        <ul>
            <li><strong>Acción Inmediata:</strong> Protocolo de confirmación diagnóstica urgente.</li>
            <li>Descartar síntomas de cetoacidosis.</li>
            <li>No esperar a pruebas adicionales para iniciar valoración.</li>
        </ul>
    </div>
    <div class="matrix-priority" style="color: {URGENT_RED}; background-color: #FFF5F5;">
        URGENTE 🚨
    </div>
</div>
<div class="matrix-row {style_alto}" style="border-left: 5px solid {CEMP_PINK};">
    <div class="matrix-result">
        <h4>Riesgo Alto Confirmado</h4>
        <p>Probabilidad IA superior al umbral (> {threshold}) y alejada de la zona de corte.</p>
    </div>
    <div class="matrix-action">
        <ul>
            <li>Derivación a Endocrinología.</li>
            <li>Solicitar HbA1c confirmatoria y perfil lipídico completo.</li>
            <li>Valorar inicio de intervención farmacológica (ej. Metformina) y nutricional intensiva.</li>
        </ul>
    </div>
    <div class="matrix-priority" style="color: {CEMP_PINK}; background-color: #FFF0F2;">
        ALTA 🔴
    </div>
</div>
<div class="matrix-row {style_incertidumbre}" style="border-left: 5px solid #F39C12;">
    <div class="matrix-result">
        <h4>Zona de Incertidumbre Clínica</h4>
        <p>Probabilidad cercana al umbral de decisión. Fiabilidad de la predicción reducida.</p>
    </div>
    <div class="matrix-action">
        <ul>
            <li>Repetir Test de Tolerancia Oral a la Glucosa (TTOG) en 3-6 meses.</li>
            <li>Monitorización más estrecha de glucemia basal y HbA1c.</li>
            <li>Refuerzo de educación diabetológica (dieta y ejercicio).</li>
        </ul>
    </div>
    <div class="matrix-priority" style="color: #F39C12; background-color: #FFF9E5;">
        MEDIA 🟠
    </div>
</div>
<div class="matrix-row {style_bajo}" style="border-left: 5px solid {GOOD_TEAL};">
    <div class="matrix-result">
        <h4>Riesgo Bajo</h4>
        <p>Probabilidad IA inferior al umbral (< {threshold}). Perfil actual no sugiere riesgo inminente.</p>
    </div>
    <div class="matrix-action">
        <ul>
            <li>Mantenimiento de seguimiento rutinario (según guías locales, ej. anual).</li>
            <li>Consejo breve sobre estilos de vida saludables.</li>
            <li>Mantener IMC < 25.</li>
        </ul>
    </div>
    <div class="matrix-priority" style="color: {GOOD_TEAL}; background-color: #F0FDF4;">
        BAJA 🟢
    </div>
</div>
</div>
""", unsafe_allow_html=True)

    with tab4:
        st.write("")
        
        c_tech_1, c_tech_2 = st.columns([1.3, 1], gap="medium")
        
        with c_tech_1:
            st.markdown(f"""
            <div class="card">
                <div class="tech-card-title">Especificaciones del Modelo</div>
                <p style="font-size:0.9rem; color:#666; margin-bottom:15px; text-align: justify;">
                    El núcleo del sistema es un algoritmo de <strong>Random Forest Classifier</strong> optimizado para datos médicos desbalanceados.
                </p>
                <div style="background:#F8F9FA; padding:15px; border-radius:8px; border:1px solid #eee; font-family:monospace; font-size:0.85rem; color:#444;">
                    <strong>Pipeline:</strong><br>
                    1. Imputer (Median)<br>
                    2. StandardScaler<br>
                    3. RandomForest (n=200, depth=5, balanced)
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card">
                <div id="metrics-anchor" class="tech-card-title">Métricas de Rendimiento (Test)</div>
                <p style="font-size:0.9rem; color:#666; margin-bottom:15px; text-align: justify;">
                    Evaluación sobre conjunto de test independiente (10 repeticiones). Se prioriza la <strong>Sensibilidad (Recall)</strong> para minimizar falsos negativos.
                </p>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th></th>
                            <th><span class="badge-standard">Umbral Estándar (0.5)</span></th>
                            <th><span class="badge-optimal">Umbral Óptimo (0.27)</span></th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="metric-name-col">Accuracy</td>
                            <td>0.738</td>
                            <td>0.719</td>
                        </tr>
                        <tr>
                            <td class="metric-name-col">Precision</td>
                            <td>0.604</td>
                            <td>0.560</td>
                        </tr>
                        <tr class="highlight-row">
                            <td class="metric-name-col" style="color:{CEMP_DARK}; font-weight:800;">Recall (Sensibilidad)</td>
                            <td>0.733</td>
                            <td class="highlight-optimal">0.924</td>
                        </tr>
                        <tr class="highlight-row">
                            <td class="metric-name-col" style="color:{CEMP_DARK}; font-weight:800;">F2-Score</td>
                            <td>0.703</td>
                            <td class="highlight-optimal">0.818</td>
                        </tr>
                        <tr>
                            <td class="metric-name-col">AUC-ROC</td>
                            <td>0.815</td>
                            <td>0.815</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with c_tech_2:
            st.markdown(f"""<div class="card" style="height:100%;">
    <div class="tech-card-title">Origen de los Datos</div>
    <p style="font-size:0.9rem; color:#666; margin-bottom: 10px; text-align: justify;">
        <strong>Fuente:</strong> Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales (NIDDK).
    </p>
    <p style="font-size:0.9rem; color:#666; margin-bottom: 15px; text-align: justify;">
        <strong>Población:</strong> Mujeres de al menos 21 años de ascendencia indígena Pima.
    </p>
    <div style="background-color: rgba(233, 127, 135, 0.1); border-left: 4px solid {CEMP_PINK}; padding: 15px; margin: 15px 0; border-radius: 8px;">
        <p style="font-size:0.85rem; color: #C0392B; margin: 0; line-height:1.5; text-align: justify;">
            ⚠️ <strong>Limitaciones del Modelo:</strong> Este prototipo se basa en un conjunto de datos histórico y reducido (Pima Indians). Los resultados reflejan patrones específicos de esta demografía y no deben generalizarse a la población global sin recalibración.
        </p>
    </div>
    <hr style="margin:20px 0; border:0; border-top:1px solid #eee;">
    <div class="tech-card-title" style="margin-bottom: 10px;">Variables Utilizadas (10)</div>
    <p style="font-size:0.85rem; font-weight:700; color:#555; margin-bottom:5px;">CLÍNICAS DIRECTAS:</p>
    <ul style="font-size:0.85rem; color:#555; padding-left:20px; line-height:1.6; margin-bottom:15px; list-style-type: disc;">
        <li><strong>Pregnancies:</strong> Nº de embarazos.</li>
        <li><strong>Glucose:</strong> Glucosa plasmática a las 2h.</li>
        <li><strong>BloodPressure:</strong> Presión arterial diastólica.</li>
        <li><strong>Insulin:</strong> Insulina sérica a las 2h.</li>
        <li><strong>BMI:</strong> Índice de Masa Corporal.</li>
        <li><strong>DPF:</strong> Predisposición genética (función pedigrí).</li>
        <li><strong>Age:</strong> Edad (años).</li>
    </ul>
    <p style="font-size:0.85rem; font-weight:700; color:{CEMP_PINK}; margin-bottom:5px;">CALCULADAS:</p>
    <ul style="font-size:0.85rem; color:#555; padding-left:20px; line-height:1.6; margin: 0; list-style-type: disc;">
        <li><strong>Indice_resistencia:</strong> Producto Glucosa × Insulina.</li>
        <li><strong>BMI_square:</strong> BMI al cuadrado (no linealidad).</li>
        <li><strong>Is_prediabetes:</strong> Indicador binario si Glucosa ≥ 140.</li>
    </ul>
</div>""", unsafe_allow_html=True)

    # FOOTER AL FINAL, GRIS Y NEUTRO
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-top: 30px; text-align: center; border: 1px solid #e0e0e0; color: #666;">
        <p style="font-size: 0.8rem; margin: 0; line-height: 1.6;">
            © 2025 <strong>Nerea Moreno Escamilla</strong><br>
            TFM • Máster en Aplicaciones de la Inteligencia Artificial en la Sanidad<br>
            CEMP - Centro Europeo de Másteres y Posgrados
        </p>
    </div>
    """, unsafe_allow_html=True)
