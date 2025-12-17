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

# 3. CSS "NUCLEAR" (Para forzar los estilos sí o sí)
st.markdown("""
<style>
    /* --- ESTRUCTURA DE FONDO Y TARJETA --- */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Ocultamos menú hamburguesa y footer */
    #MainMenu, footer, header {visibility: hidden;}

    /* LA TARJETA BLANCA (Contenedor Principal) */
    /* Forzamos padding y centrado */
    .block-container {
        background-color: white;
        padding: 3rem 2rem !important;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        max-width: 700px;
        margin-top: 2rem;
    }

    /* --- TÍTULO --- */
    h1 {
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-weight: 900 !important;
        font-size: 3rem !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
        color: #2c3e50;
    }

    /* Clases de color para el título */
    .txt-dark { color: #2c3e50; }
    .txt-pink { color: #ef7d86; }
    .txt-gray { color: #bdc3c7; }

    /* --- TEXTOS VARIOS --- */
    .badge-container { 
        display: flex; 
        justify-content: center; 
        margin-bottom: 15px; 
    }
    
    .badge {
        background-color: #2c3e50;
        color: white;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .institution {
        text-align: center;
        color: #555;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #34495e;
        margin-top: 10px;
        margin-bottom: 25px;
    }

    .description {
        text-align: center;
        color: #666;
        line-height: 1.6;
        font-size: 0.95rem;
        margin-bottom: 30px;
        padding: 0 10px;
    }

    /* --- CAJA DE ADVERTENCIA --- */
    .warning-box {
        background-color: #f9fafb;
        border-left: 5px solid #ef7d86;
        padding: 20px;
        border-radius: 5px;
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 30px;
        text-align: center; /* Texto centrado dentro de la caja */
    }
    
    /* Negrita forzada */
    .bold-text {
        font-weight: 900 !important; /* Extra negrita */
        color: #444;
        display: block;
        margin-bottom: 8px;
    }

    /* --- LA CLAVE PARA CENTRAR EL BOTÓN --- */
    /* Esto afecta al contenedor div que envuelve al botón en Streamlit */
    div.stButton {
        display: flex !important;
        flex-direction: row;
        justify-content: center !important; /* EJE HORIZONTAL: CENTRO */
        width: 100% !important;
    }

    /* Estilo visual del botón */
    div.stButton > button {
        background: linear-gradient(90deg, #ef707a 0%, #e8aeb3 100%);
        color: white !important;
        border: none;
        padding: 14px 40px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 6px 20px rgba(239, 112, 122, 0.3);
        transition: all 0.3s ease;
        
        /* IMPORTANTE: Evita que el botón se estire */
        width: auto !important; 
        display: inline-block !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(239, 112, 122, 0.5);
    }
    
    /* Quitamos borde rojo al hacer click */
    div.stButton > button:focus:not(:focus-visible) {
        color: white !important;
        border-color: transparent !important;
        outline: none;
    }

</style>
""", unsafe_allow_html=True)

# 4. Vistas
if st.session_state.page == 'landing':
    
    # Renderizado HTML (Título, textos y advertencia)
    st.markdown("""
        <div class="badge-container">
            <span class="badge">TFM • Máster en Inteligencia Artificial aplicada a la salud</span>
        </div>
        
        <div class="institution">Centro Europeo de Másteres y Posgrados</div>
        
        <h1>
            <span class="txt-dark">D</span><span class="txt-pink">IA</span><span class="txt-dark">BETES</span><span class="txt-gray">.</span><span class="txt-pink">NME</span>
        </h1>
        
        <div class="subtitle">Prototipo de CDSS para el diagnóstico temprano de diabetes</div>
        
        <p class="description">
            Este proyecto explora el potencial de integrar modelos predictivos avanzados en el flujo de trabajo 
            clínico, visualizando un futuro donde la IA actúa como un potente aliado en la detección temprana y 
            prevención de la diabetes tipo 2.
        </p>

        <div class="warning-box">
            <span class="bold-text">Aplicación desarrollada con fines exclusivamente educativos como parte de un Trabajo de Fin de Máster.</span>
            
            <span>⚠️ Esta herramienta NO es un dispositivo médico certificado. Los resultados son una simulación académica y NO deben 
            utilizarse para el diagnóstico real, tratamiento o toma de decisiones clínicas.</span>
        </div>
    """, unsafe_allow_html=True)

    # BOTÓN (Ahora está obligado a centrarse por el CSS 'div.stButton')
    if st.button("INICIAR SIMULACIÓN ➔"):
        ir_a_simulacion()
        st.rerun()

elif st.session_state.page == 'simulacion':
    # --- PANTALLA SECUNDARIA ---
    if st.button("⬅ Volver"):
        volver_inicio()
        st.rerun()

    st.title("Panel de Diagnóstico Clínico")
    
    with st.form("my_form"):
        st.write("Datos del paciente:")
        c1, c2 = st.columns(2)
        c1.number_input("Edad")
        c2.number_input("Glucosa")
        submitted = st.form_submit_button("Predecir")
