import streamlit as st

# 1. Configuración de página
st.set_page_config(
    page_title="Diabetes NME",
    page_icon="🩸",
    layout="centered"
)

# 2. Gestión de navegación
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def ir_a_simulacion():
    st.session_state.page = 'simulacion'

def volver_inicio():
    st.session_state.page = 'landing'

# 3. CSS personalizado
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        background-color: white;
        padding: 3rem !important;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        max-width: 800px;
        margin-top: 2rem;
    }

    h1 {
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        color: #2c3e50;
        margin-bottom: 0 !important;
        line-height: 1.2 !important;
    }

    .landing-pink { color: #ef7d86; }
    .landing-gray { color: #bdc3c7; }

    .badge-container { text-align: center; margin-bottom: 10px; }

    .badge {
        background-color: #2c3e50;
        color: white;
        padding: 6px 15px;
        border-radius: 50px;
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
    }

    .institution {
        text-align: center;
        color: #555;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #34495e;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .description {
        text-align: center;
        color: #666;
        line-height: 1.6;
        font-size: 0.95rem;
        margin-bottom: 30px;
        padding: 0 20px;
    }

    .warning-box {
        background-color: #f9fafb;
        border-left: 4px solid #ef7d86;
        padding: 20px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 30px;
        text-align: center;
    }

    .warning-box p {
        margin: 0;
        line-height: 1.5;
    }

    /* CONTENEDOR DEL BOTÓN */
    .center-button {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    /* BOTÓN */
    div.stButton > button {
        background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%);
        color: white;
        border: none;
        padding: 12px 50px;
        border-radius: 50px;
        font-weight: bold;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(239, 112, 122, 0.3);
        transition: all 0.3s ease;
        width: auto;
        white-space: nowrap;   /* ⬅️ CLAVE */
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(239, 112, 122, 0.5);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 4. Vistas
if st.session_state.page == 'landing':

    st.markdown("""
        <div class="badge-container">
            <span class="badge">TFM • Máster en Inteligencia Artificial aplicada a la salud</span>
        </div>

        <div class="institution">Centro Europeo de Másteres y Posgrados</div>

        <h1>D<span class="landing-pink">IA</span>BETES<span class="landing-gray">.</span><span class="landing-pink">NME</span></h1>

        <div class="subtitle">Prototipo de CDSS para el diagnóstico temprano de diabetes</div>

        <p class="description">
            Este proyecto explora el potencial de integrar modelos predictivos avanzados en el flujo de trabajo 
            clínico, visualizando un futuro donde la IA actúa como un potente aliado en la detección temprana y 
            prevención de la diabetes tipo 2.
        </p>

        <div class="warning-box">
            <p><strong>Aplicación desarrollada con fines exclusivamente educativos como parte de un Trabajo de Fin de Máster.</strong></p>
            <p style="margin-top:10px;">
                ⚠️ Esta herramienta NO es un dispositivo médico certificado. Los resultados son una simulación académica y NO deben 
                utilizarse para el diagnóstico real, tratamiento o toma de decisiones clínicas.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # BOTÓN CENTRADO Y EN UNA SOLA LÍNEA
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button("INICIAR SIMULACIÓN ➔"):
        ir_a_simulacion()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'simulacion':

    if st.button("⬅ Volver"):
        volver_inicio()
        st.rerun()

    st.title("Panel de Diagnóstico Clínico")

    with st.form("patient_data"):
        st.write("Introduzca los datos del paciente:")
        c1, c2 = st.columns(2)
        c1.number_input("Edad", step=1)
        c2.number_input("Glucosa (mg/dL)")
        c1.number_input("IMC")
        c2.selectbox("Antecedentes", ["Sí", "No"])

        submitted = st.form_submit_button("Ejecutar Predicción")
        if submitted:
            st.success("Procesando...")
