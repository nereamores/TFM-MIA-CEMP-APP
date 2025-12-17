import streamlit as st

# =========================================================
# Configuración de página
# =========================================================
st.set_page_config(
    page_title="Diabetes NME",
    page_icon="🩸",
    layout="centered"
)

# =========================================================
# Estado de navegación
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "landing"

def ir_a_simulacion():
    st.session_state.page = "simulacion"

def volver_inicio():
    st.session_state.page = "landing"

# =========================================================
# CSS GLOBAL (incluye centrado óptico del texto del botón)
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

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
        font-family: Arial, sans-serif;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        color: #2c3e50;
        margin-bottom: 0 !important;
        line-height: 1.2 !important;
    }

    .landing-pink { color: #ef7d86; }
    .landing-gray { color: #bdc3c7; }

    .badge-container {
        text-align: center;
        margin-bottom: 10px;
    }

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

    /* -------------------------
       BOTÓN: estructura principal
       ------------------------- */
    div.stButton > button {
        position: relative;              /* para posicionar el span y la flecha */
        background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%);
        color: white;
        border: none;
        padding: 14px 80px;              /* padding lateral amplio para que la caja tenga forma redondeada */
        border-radius: 50px;
        font-weight: bold;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        white-space: nowrap;
        box-shadow: 0 4px 15px rgba(239,112,122,0.3);
        cursor: pointer;
        overflow: visible;               /* que no corte el pseudo elemento */
    }

    /* Hacemos el texto del botón un elemento posicionado para centrarlo ópticamente */
    div.stButton > button > span {
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%) translateX(4px); /* <-- ajuste óptico: 4px a la derecha */
        display: inline-block;
        pointer-events: none; /* que el click vaya al botón */
    }

    /* Flecha decorativa en el lado derecho */
    div.stButton > button::after {
        content: "➔";
        position: absolute;
        right: 28px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 16px;
        pointer-events: none;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(239,112,122,0.5);
        color: white;
    }

    /* Pequeña mejora responsiva: reducir padding en pantallas estrechas */
    @media (max-width: 600px) {
        div.stButton > button {
            padding: 12px 40px;
            font-size: 13px;
        }
        div.stButton > button::after {
            right: 18px;
        }
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LANDING PAGE
# =========================================================
if st.session_state.page == "landing":

    st.markdown("""
        <div class="badge-container">
            <span class="badge">TFM • Máster en Inteligencia Artificial aplicada a la salud</span>
        </div>

        <div class="institution">Centro Europeo de Másteres y Posgrados</div>

        <h1>
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
                diagnóstico real, tratamiento o toma de decisiones clínicas.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # -------- BOTÓN CENTRADO (COLUMNAS) --------
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Observa: mantenemos st.button para que Streamlit maneje el estado correctamente
        if st.button("INICIAR SIMULACIÓN"):
            ir_a_simulacion()
            st.rerun()

# =========================================================
# PÁGINA DE SIMULACIÓN
# =========================================================
elif st.session_state.page == "simulacion":

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
