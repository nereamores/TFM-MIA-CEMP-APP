import streamlit as st

# 1. Configuración de página
st.set_page_config(
    page_title="Diabetes NME",
    page_icon="🩸",
    layout="centered"
)

# 2. Gestión de navegación (Estado)
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def ir_a_simulacion():
    st.session_state.page = 'simulacion'

def volver_inicio():
    st.session_state.page = 'landing'

# 3. CSS A MEDIDA (centrado absoluto del botón dentro del card)
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }

    /* ocultar cabeceras por defecto */
    #MainMenu, footer, header {visibility: hidden;}

    /* Contenedor principal (cuadro blanco) */
    .block-container {
        background-color: white;
        /* aumentamos padding-bottom para dejar espacio al botón absoluto */
        padding: 3rem 3rem 5.5rem 3rem !important;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        max-width: 800px;
        margin-top: 2rem;
        position: relative; /* <-- necesario para posicionar el botón absolutamente dentro */
    }

    /* Título */
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

    /* ===== Posicionamiento absoluto y centrado real del botón dentro del card ===== */
    /* Esto sitúa el botón en el eje central del .block-container y lo coloca hacia el final (bottom) */
    .block-container div.stButton {
        position: absolute !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        bottom: 24px !important; /* distancia desde la base del cuadro blanco */
        width: auto !important;
        display: flex !important;
        justify-content: center !important;
        padding: 0 !important;
    }

    /* Estilo visual del botón */
    .block-container div.stButton > button {
        display: inline-block;
        background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 50px !important;
        border-radius: 50px !important;
        font-weight: bold !important;
        font-size: 14px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 15px rgba(239, 112, 122, 0.3) !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important; /* evitar quiebre de línea */
        cursor: pointer !important;
    }

    .block-container div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(239,112,122,0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# 4. Renderizado de Vistas (texto EXACTO conservado)
if st.session_state.page == 'landing':
    
    # HTML Estático (texto intacto)
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
            <p style="margin-bottom: 10px;">
                <strong>Aplicación desarrollada con fines exclusivamente educativos como parte de un Trabajo de Fin de Máster.</strong>
            </p>
            
            <p>
                ⚠️ Esta herramienta NO es un dispositivo médico certificado. Los resultados son una simulación académica y NO deben 
                utilizarse para el diagnóstico real, tratamiento o toma de decisiones clínicas.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # BOTÓN STREAMLIT (estará posicionado absolutamente y centrado respecto al .block-container)
    if st.button("INICIAR SIMULACIÓN ➔"):
        ir_a_simulacion()
        st.rerun()

elif st.session_state.page == 'simulacion':
    # --- PANTALLA DE SIMULACIÓN ---
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
