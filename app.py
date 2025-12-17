import streamlit as st

# --------------------------------------------------
# 1. Configuración de página (DEBE IR LO PRIMERO)
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes NME",
    page_icon="🩸",
    layout="centered"
)

# --------------------------------------------------
# 2. Gestión de navegación (estado)
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

def ir_a_simulacion():
    st.session_state.page = "simulacion"

def volver_inicio():
    st.session_state.page = "landing"

# --------------------------------------------------
# 3. CSS personalizado
# --------------------------------------------------
st.markdown(
    """
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
        font-weight: 900 !important;
        font-size: 3.2rem !important;
        color: #2c3e50;
        margin-bottom: 0;
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
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
        color: #555;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 25px;
        color: #34495e;
    }

    .description {
        text-align: center;
        font-size: 0.95rem;
        color: #666;
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
        text-align: center;
        margin-bottom: 25px;
    }

    div.stButton {
        display: flex;
        justify-content: center;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%);
        color: white;
        border: none;
        padding: 12px 50px;
        border-radius: 50px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(239,112,122,0.3);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(239,112,122,0.5);
    }
</style>
""",
    unsafe_allow_html=True
)

# --------------------------------------------------
# 4. Vistas
# --------------------------------------------------
if st.session_state.page == "landing":

    st.markdown(
        """
<div class="badge-container">
    <span class="badge">TFM • Máster en IA aplicada a la salud</span>
</div>

<div class="institution">Centro Europeo de Másteres y Posgrados</div>

<h1>D<span class="landing-pink">IA</span>BETES<span class="landing-gray">.</span><span class="landing-pink">NME</span></h1>

<div class="subtitle">
    Prototipo de CDSS para el diagnóstico temprano de diabetes
</div>

<p class="description">
    Proyecto académico que explora el uso de modelos predictivos avanzados
    integrados en el flujo clínico para apoyar la detección temprana
    de la diabetes tipo 2.
</p>

<div class="warning-box">
    <strong>Aplicación desarrollada exclusivamente con fines educativos.</strong><br><br>
    ⚠️ No es un dispositivo médico certificado y no debe utilizarse para
    decisiones clínicas reales.
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button("Iniciar simulación ➜"):
        ir_a_simulacion()
        st.rerun()

# --------------------------------------------------
elif st.session_state.page == "simulacion":

    if st.button("⬅ Volver"):
        volver_inicio()
        st.rerun()

    st.title("Panel de Diagnóstico Clínico")

    with st.form("patient_data"):
        st.write("Introduzca los datos del paciente:")

        c1, c2 = st.columns(2)

        edad = c1.number_input("Edad", min_value=0, max_value=120, step=1, key="edad")
        glucosa = c2.number_input("Glucosa (mg/dL)", min_value=0.0, key="glucosa")
        imc = c1.number_input("IMC", min_value=0.0, key="imc")
        antecedentes = c2.selectbox("Antecedentes familiares", ["No", "Sí"], key="ant")

        submitted = st.form_submit_button("Ejecutar predicción")

        if submitted:
            st.success("Procesando predicción (simulación académica)...")
