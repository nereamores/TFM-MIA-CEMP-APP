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

# 3. CSS "Pixel-Perfect" para replicar tu diseño
st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background-color: #f0f2f6;
    }

    /* Ocultar elementos nativos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* LA TARJETA BLANCA (Contenedor Principal) */
    .block-container {
        background-color: white;
        padding: 3rem !important;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        max-width: 800px;
        margin-top: 2rem;
    }

    /* TIPOGRAFÍA Y TEXTOS */
    h1 {
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        color: #2c3e50;
        margin-bottom: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }

    /* Clase para la parte rosada del logo */
    .logo-highlight {
        color: #ef7d86;
    }

    /* Badge superior negro */
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

    /* Institución (Texto gris mayúsculas) */
    .institution {
        text-align: center;
        color: #555;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    /* Subtítulo */
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #34495e;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    /* Párrafo descriptivo */
    .description {
        text-align: center;
        color: #666;
        line-height: 1.6;
        font-size: 0.95rem;
        margin-bottom: 30px;
        padding: 0 20px;
    }

    /* CAJA DE ADVERTENCIA COMPLETA */
    .warning-box {
        background-color: #f9fafb;
        border-left: 4px solid #ef7d86; /* Borde rosa a la izquierda */
        padding: 20px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 30px; /* Espacio antes del botón */
        text-align: center;
    }
    
    .warning-text-sm {
        display: block;
        margin-bottom: 10px;
        color: #777;
    }

    /* ESTILO DEL BOTÓN (CENTRADO PERFECTO) */
    /* Esto centra el contenedor del botón */
    div.stButton {
        text-align: center; 
    }

    /* Esto da estilo al botón en sí */
    div.stButton > button {
        background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%);
        color: white;
        border: none;
        padding: 12px 40px;
        border-radius: 50px;
        font-weight: bold;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(239, 112, 122, 0.3);
        transition: all 0.3s ease;
        
        /* Truco para centrar elemento block */
        display: block; 
        margin: 0 auto; 
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(239, 112, 122, 0.5);
        color: white;
        border-color: transparent;
    }
    
    div.stButton > button:active {
        color: white;
        border-color: transparent;
    }

</style>
""", unsafe_allow_html=True)

# 4. Renderizado de Vistas
if st.session_state.page == 'landing':
    
    # HTML Estático para textos y logos
    st.markdown("""
        <div class="badge-container">
            <span class="badge">TFM • Máster en Inteligencia Artificial aplicada a la salud</span>
        </div>
        
        <div class="institution">Centro Europeo de Másteres y Posgrados</div>
        
        <h1>DIABETES<span class="logo-highlight">.NME</span></h1>
        
        <div class="subtitle">Prototipo de CDSS para el diagnóstico temprano de diabetes</div>
        
        <p class="description">
            Este proyecto explora el potencial de integrar modelos predictivos avanzados en el flujo de trabajo 
            clínico, visualizando un futuro donde la IA actúa como un potente aliado en la detección temprana y 
            prevención de la diabetes tipo 2.
        </p>

        <div class="warning-box">
            <span class="warning-text-sm">Aplicación desarrollada con fines exclusivamente educativos como parte de un Trabajo de Fin de Máster.</span>
            <strong>⚠️ Esta herramienta NO es un dispositivo médico certificado.</strong> Los resultados son una simulación académica y NO deben 
            utilizarse para el diagnóstico real, tratamiento o toma de decisiones clínicas.
        </div>
    """, unsafe_allow_html=True)

    # BOTÓN DE STREAMLIT (Ahora centrado por CSS)
    if st.button("INICIAR SIMULACIÓN ➔"):
        ir_a_simulacion()
        st.rerun()

elif st.session_state.page == 'simulacion':
    # --- TU CÓDIGO DE LA APP DE PREDICCIÓN VA AQUÍ ---
    
    # Botón pequeño para volver (sin estilo fancy para diferenciarlo)
    if st.button("⬅ Volver"):
        volver_inicio()
        st.rerun()

    st.title("Panel de Diagnóstico Clínico")
    st.info("El modelo predictivo está listo para recibir datos.")

    # Ejemplo de formulario
    with st.form("patient_data"):
        c1, c2 = st.columns(2)
        c1.text_input("ID Paciente")
        c2.number_input("Edad", step=1)
        c1.number_input("Glucosa en sangre (mg/dL)")
        c2.number_input("IMC (Índice de Masa Corporal)")
        
        submitted = st.form_submit_button("Ejecutar Predicción")
        if submitted:
            st.success("Analizando datos...")
