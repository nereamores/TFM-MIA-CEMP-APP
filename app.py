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
        if isinstance(X, pd.DataFrame):
            score = (X.iloc[0, 1]*0.5) + (X.iloc[0, 4]*0.4) + (X.iloc[0, 6]*0.1) 
        else:
            score = 50
        prob = 1 / (1 + np.exp(-(score - 100) / 15)) 
        return [[1-prob, prob]]

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

if 'model' not in st.session_state:
    st.session_state.model = load_model()

if 'predict_clicked' not in st.session_state:
    st.session_state.predict_clicked = False

# =========================================================
# 2. FUNCIONES AUXILIARES (VISUALIZACIÓN)
# =========================================================

# Esta función es CLAVE para meter la imagen DENTRO de la tarjeta HTML
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=150) # DPI ajustado para web
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def get_help_icon(description):
    return f"""<span style="display:inline-block; width:16px; height:16px; line-height:16px; text-align:center; border-radius:50%; background:#E0E0E0; color:#777; font-size:0.7rem; font-weight:bold; cursor:help; margin-left:6px; position:relative; top:-1px;" title="{description}">?</span>"""

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
            color: #2c3e50; margin-bottom: 0 !important;
            line-height: 1.2 !important; letter-spacing: -1px; cursor: default;
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
        div.stButton > button {
            background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%);
            color: white; border: none; padding: 14px 80px;
            border-radius: 50px; font-weight: bold; font-size: 14px;
            text-transform: uppercase; cursor: pointer;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
<div style="text-align: center; margin-bottom: 10px;">
<span style="background-color: #2c3e50; color: white; padding: 6px 15px; border-radius: 50px; font-size: 11px; font-weight: bold; text-transform: uppercase;">TFM • Máster en Inteligencia Artificial aplicada a la salud</span>
</div>
<div style="text-align: center; color: #555; font-size: 13px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px;">Centro Europeo de Másteres y Posgrados</div>
<h1>D<span style="color: #ef7d86;">IA</span>BETES<span style="color: #bdc3c7;">.</span><span style="color: #ef7d86;">NME</span></h1>
<div class="subtitle">Prototipo de CDSS para el diagnóstico temprano de diabetes</div>
<p class="description">Este proyecto explora el potencial de integrar modelos predictivos avanzados en el flujo de trabajo clínico.</p>
<div class="warning-box">
    <p><strong>Aplicación con fines educativos.</strong> ⚠️ NO es un dispositivo médico certificado.</p>
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
    
    BMI_GRADIENT = "linear-gradient(90deg, #81D4FA 0%, #4DB6AC 25%, #FFF176 40%, #FFB74D 55%, #E97F87 70%, #880E4F 100%)"
    GLUCOSE_GRADIENT = "linear-gradient(90deg, #4DB6AC 0%, #4DB6AC 28%, #FFF176 32%, #FFB74D 48%, #E97F87 52%, #880E4F 100%)"

    st.markdown(f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .block-container {{ max-width: 1250px; padding-top: 2rem; }}
        .cemp-logo {{ font-family: 'Helvetica', sans-serif; font-weight: 800; font-size: 1.8rem; color: {CEMP_DARK}; margin: 0; }}
        .cemp-logo span {{ color: {CEMP_PINK}; }}
        .card {{
            background-color: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.04);
            margin-bottom: 15px; display: flex; flex-direction: column; justify-content: center;
        }}
        .card-auto {{ height: 100%; }}
        .card-header {{ color: #999; font-size: 0.75rem; font-weight: bold; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 15px; }}
        .calc-box {{ background-color: #F8F9FA; border-radius: 8px; padding: 12px 15px; border: 1px solid #EEE; margin-top: 5px; margin-bottom: 20px; }}
        .calc-label {{ font-size: 0.75rem; color: #888; font-weight: 600; text-transform: uppercase; }}
        .calc-value {{ font-size: 1rem; color: {CEMP_DARK}; font-weight: 800; }}
        
        /* Estilos para las barras */
        .bar-container {{ position: relative; width: 100%; margin-top: 20px; margin-bottom: 30px; }}
        .bar-bg {{ background: #F0F2F5; height: 12px; border-radius: 6px; width: 100%; overflow: hidden; }}
        .bar-fill {{ height: 100%; width: 100%; border-radius: 6px; }}
        .bar-marker {{ position: absolute; top: -6px; width: 4px; height: 24px; background: {CEMP_DARK}; border: 1px solid white; z-index: 10; }}
        .bar-txt {{ position: absolute; top: -30px; transform: translateX(-50%); font-size: 0.85rem; font-weight: bold; color: {CEMP_DARK}; background: white; padding: 2px 8px; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .legend-container {{ position: relative; width: 100%; height: 20px; margin-top: 8px; }}
        .legend-label {{ position: absolute; transform: translateX(-50%); font-size: 0.7rem; color: #888; font-weight: 600; }}
        </style>
    """, unsafe_allow_html=True)

    def input_biomarker(label_text, min_val, max_val, default_val, key, help_text="", format_str=None):
        st.markdown(f"**{label_text}**" + (get_help_icon(help_text) if help_text else ""), unsafe_allow_html=True)
        c1, c2 = st.columns([2.5, 1], gap="small")
        
        if key not in st.session_state: st.session_state[key] = default_val
        
        def update(): st.session_state.predict_clicked = False
        
        with c1:
            st.slider("", min_val, max_val, key=f"{key}_slider", value=st.session_state[key], on_change=lambda: [st.session_state.update({key: st.session_state[f"{key}_slider"]}), update()], label_visibility="collapsed")
        with c2:
            st.number_input("", min_val, max_val, key=f"{key}_input", value=st.session_state[key], on_change=lambda: [st.session_state.update({key: st.session_state[f"{key}_input"]}), update()], label_visibility="collapsed", format=format_str)
        
        return st.session_state[key]

    with st.sidebar:
        if st.button("⬅ Volver"):
            volver_inicio()
            st.rerun()

        st.markdown(f'<div class="cemp-logo">D<span>IA</span>BETES<span style="color:#BDC3C7">.</span><span>NME</span></div>', unsafe_allow_html=True)
        st.caption("CLINICAL DECISION SUPPORT SYSTEM")
        
        st.markdown("---")
        
        def reset(): st.session_state.predict_clicked = False
        patient_name = st.text_input("ID Paciente", value="Paciente #8842-X", on_change=reset)
        consult_date = st.date_input("Fecha Predicción", value=datetime.date.today(), on_change=reset)
        date_str = f"{consult_date.day}/{consult_date.month}/{consult_date.year}"

        st.markdown("---")
        
        glucose = input_biomarker("Glucosa 2h (mg/dL)", 50, 350, 50, "gluc", format_str="%d")
        insulin = input_biomarker("Insulina (µU/ml)", 0, 900, 0, "ins", format_str="%d")
        blood_pressure = input_biomarker("Presión Arterial (mm Hg)", 0, 150, 0, "bp", format_str="%d")
        
        proxy_index = int(glucose * insulin)
        st.markdown(f"""<div class="calc-box" style="border-left: 4px solid {CEMP_PINK};"><div style="display:flex; justify-content:space-between;"><span class="calc-label">Índice RI</span><span class="calc-value">{proxy_index}</span></div></div>""", unsafe_allow_html=True)

        st.markdown("---") 

        weight = input_biomarker("Peso (kg)", 30.0, 250.0, 30.0, "weight")
        height = input_biomarker("Altura (m)", 1.00, 2.20, 1.00, "height")
        
        bmi = weight / (height * height) if height > 0 else 0
        bmi_sq = bmi ** 2
        st.markdown(f"""<div class="calc-box" style="border-left: 4px solid {CEMP_PINK};"><div style="display:flex; justify-content:space-between;"><span class="calc-label">BMI (kg/m²)</span><span class="calc-value">{bmi:.2f}</span></div></div>""", unsafe_allow_html=True)
        
        st.markdown("---") 

        c1, c2 = st.columns(2)
        with c1: age = input_biomarker("Edad", 18, 90, 18, "age", format_str="%d")
        with c2: pregnancies = input_biomarker("Embarazos", 0, 20, 0, "preg", format_str="%d")
        
        st.markdown("---") 
        dpf = input_biomarker("Ant. Familiares (DPF)", 0.0, 2.5, 0.0, "dpf")

    # --- CONTENIDO PRINCIPAL ---
    st.markdown(f"<h1 style='color:{CEMP_DARK}; font-size: 2.2rem;'>Evaluación de Riesgo Diabético</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Panel General", "Explicabilidad", "Protocolo"])

    # PREPARACIÓN DE DATOS
    threshold = 0.27
    is_prediabetes = 1 if glucose >= 140 else 0
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, insulin, bmi, dpf, age, proxy_index, bmi_sq, is_prediabetes]], 
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DPF', 'Age', 'Indice_resistencia', 'BMI_square', 'Is_prediabetes'])
    
    if hasattr(st.session_state.model, 'predict_proba'):
        try:
            prob = st.session_state.model.predict_proba(input_data)[0][1]
        except:
            prob = 0.5
    else:
        prob = 0.5

    is_high = prob > threshold
    risk_label = "ALTO RIESGO" if is_high else "BAJO RIESGO"
    risk_color = CEMP_PINK if is_high else GOOD_TEAL
    
    with tab1:
        st.write("")
        c_left, c_right = st.columns([1.8, 1], gap="medium") 
        
        with c_left:
            badges = f"""<div style="background:{'#FFF5F5' if is_high else '#F0FDF4'}; border:1px solid {risk_color}; color:{risk_color}; font-weight:bold; padding:8px 16px; border-radius:30px;">{'🔴' if is_high else '🟢'} {risk_label}</div>""" if st.session_state.predict_clicked else "<div style='color:#ccc; font-style:italic;'>Análisis pendiente...</div>"
            
            st.markdown(f"""
            <div class="card card-auto" style="flex-direction:row; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:20px;">
                    <div style="background:rgba(233, 127, 135, 0.1); width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2rem;">👤</div>
                    <div><span class="card-header">EXPEDIENTE</span><h2 style="margin:0; color:{CEMP_DARK};">{patient_name}</h2><div style="color:#666; font-size:0.9rem;">{date_str}</div></div>
                </div>
                <div>{badges}</div>
            </div>""", unsafe_allow_html=True)

            # Barras contexto
            g_pos = min(100, max(0, (glucose - 50) / 3.0)) 
            b_pos = min(100, max(0, (bmi - 10) * 2.5)) 
            st.markdown(f"""<div class="card"><span class="card-header">CONTEXTO</span>
                <div style="margin-top:15px; font-weight:bold; color:#666;">GLUCOSA ({glucose} mg/dL)</div>
                <div class="bar-container"><div class="bar-bg"><div class="bar-fill" style="background:{GLUCOSE_GRADIENT}"></div></div><div class="bar-marker" style="left:{g_pos}%"></div></div>
                <div style="margin-top:25px; font-weight:bold; color:#666;">BMI ({bmi:.1f})</div>
                <div class="bar-container"><div class="bar-bg"><div class="bar-fill" style="background:{BMI_GRADIENT}"></div></div><div class="bar-marker" style="left:{b_pos}%"></div></div>
            </div>""", unsafe_allow_html=True)

        with c_right:
            st.markdown(f"""<div class="card card-auto" style="border-left:5px solid {risk_color};"><span class="card-header" style="color:{risk_color}">ESTADO</span><h3 style="margin:0; color:{CEMP_DARK}">{'Posible Diabetes' if glucose > 126 else 'Sin hallazgos agudos'}</h3></div>""", unsafe_allow_html=True)
            
            if st.button("CALCULAR RIESGO", use_container_width=True, type="primary"):
                st.session_state.predict_clicked = True
                st.rerun()

            fig, ax = plt.subplots(figsize=(3, 3))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            if st.session_state.predict_clicked:
                ax.pie([prob, 1-prob], colors=[risk_color, '#EEE'], startangle=90, counterclock=False, wedgeprops=dict(width=0.15))
                center_text = f"{prob*100:.1f}%"
            else:
                ax.pie([100], colors=['#EEE'], wedgeprops=dict(width=0.15))
                center_text = "---"
            
            # Generamos imagen base64 para inyectar en HTML
            chart_img = fig_to_base64(fig)
            plt.close(fig)
            
            st.markdown(f"""<div class="card" style="text-align:center;">
                <span class="card-header">PROBABILIDAD IA</span>
                <div style="position:relative; width:150px; margin:0 auto;">
                    <img src="data:image/png;base64,{chart_img}" style="width:100%">
                    <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-weight:800; font-size:1.5rem; color:{CEMP_DARK};">{center_text}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.write("")
        c_exp1, c_exp2 = st.columns(2, gap="medium")
        
        with c_exp1:
            # GRÁFICO 1: IMPORTANCIA GLOBAL
            img_html_imp = ""
            if hasattr(st.session_state.model, 'named_steps'):
                try:
                    rf = st.session_state.model.named_steps['model']
                    importances = rf.feature_importances_
                    feat_names_es = ['Embarazos', 'Glucosa', 'Presión', 'Insulina', 'BMI', 'DPF', 'Edad', 'Índice R', 'BMI²', 'Prediab']
                    df_imp = pd.DataFrame({'Feature': feat_names_es, 'Imp': importances}).sort_values('Imp', ascending=True)
                    
                    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
                    ax_imp.barh(df_imp['Feature'], df_imp['Imp'], color=CEMP_PINK)
                    ax_imp.spines['top'].set_visible(False)
                    ax_imp.spines['right'].set_visible(False)
                    plt.tight_layout()
                    img_html_imp = f'<img src="data:image/png;base64,{fig_to_base64(fig_imp)}" style="width:100%">'
                    plt.close(fig_imp)
                except: pass
            
            # INYECCIÓN HTML PARA QUE ESTÉ DENTRO DEL RECUADRO
            st.markdown(f"""
            <div class="card card-auto" style="height:100%; justify-content:flex-start;">
                <span class="card-header" style="text-align:center; display:block;">VISIÓN GLOBAL DEL MODELO</span>
                {img_html_imp}
            </div>
            """, unsafe_allow_html=True)
            
            # CAJA EXPLICATIVA ROSA (FUERA DEL RECUADRO BLANCO, ABAJO)
            st.markdown(f"""
            <div style="margin-top:10px; background-color:rgba(233, 127, 135, 0.15); padding:15px; border-radius:8px; border-left:4px solid {CEMP_PINK}; color:#555; font-size:0.85rem;">
                <strong>¿Qué muestra este gráfico?</strong><br>
                Muestra qué variables son más importantes para el modelo en general. La suma de todas las barras es el 100% de la decisión. 
                Si la barra es larga, esa variable influye mucho en todos los pacientes.
            </div>
            """, unsafe_allow_html=True)

        with c_exp2:
            # GRÁFICO 2: SHAP
            img_html_shap = ""
            if SHAP_AVAILABLE and hasattr(st.session_state.model, 'named_steps'):
                try:
                    pipeline = st.session_state.model
                    step1 = pipeline.named_steps['imputer'].transform(input_data)
                    step2 = pipeline.named_steps['scaler'].transform(step1)
                    input_trans = pd.DataFrame(step2, columns=input_data.columns)
                    
                    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
                    shap_values = explainer.shap_values(input_trans)
                    
                    if isinstance(shap_values, list): sv = shap_values[1][0]
                    else: sv = shap_values[0] if len(shap_values.shape)==1 else shap_values[0,:,1]
                    
                    base = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value

                    exp = shap.Explanation(sv, base, input_data.iloc[0].values, feature_names=input_data.columns)
                    
                    fig_shap, ax_shap = plt.subplots(figsize=(5, 4))
                    shap.plots.waterfall(exp, show=False, max_display=10)
                    plt.tight_layout()
                    img_html_shap = f'<img src="data:image/png;base64,{fig_to_base64(fig_shap)}" style="width:100%">'
                    plt.close(fig_shap)
                except: pass

            # INYECCIÓN HTML
            st.markdown(f"""
            <div class="card card-auto" style="height:100%; justify-content:flex-start;">
                <span class="card-header" style="text-align:center; display:block;">ANÁLISIS INDIVIDUAL (SHAP)</span>
                {img_html_shap}
            </div>
            """, unsafe_allow_html=True)

            # CAJA EXPLICATIVA ROSA
            st.markdown(f"""
            <div style="margin-top:10px; background-color:rgba(233, 127, 135, 0.15); padding:15px; border-radius:8px; border-left:4px solid {CEMP_PINK}; color:#555; font-size:0.85rem;">
                <strong>¿Cómo se lee esto?</strong><br>
                Partimos de una probabilidad base (E[f(x)]). <br>
                <span style="color:#E97F87; font-weight:bold;">Rojo (+):</span> Factores que <strong>aumentan</strong> el riesgo.<br>
                <span style="color:#4DB6AC; font-weight:bold;">Azul (-):</span> Factores que <strong>disminuyen</strong> el riesgo.<br>
                La suma final nos da la probabilidad de diabetes de este paciente.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.write("")
        st.info("Protocolos clínicos...")
