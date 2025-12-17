import streamlit as st

# 1. Configuración de la página
st.set_page_config(
    page_title="Diabetes NME - CDSS",
    page_icon="🩸",
    layout="centered"
)

# 2. Gestión del Estado (Navegación entre la portada y la simulación)
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def iniciar_simulacion():
    st.session_state.page = 'simulacion'

def volver_inicio():
    st.session_state.page = 'landing'

# 3. Inyección de CSS (Para que se vea idéntico al diseño)
def local_css():
    st.markdown("""
    <style>
        /* Fondo gris suave de la aplicación */
        .stApp {
            background-color: #f0f2f6;
        }
        
        /* Ocultar elementos por defecto de Streamlit que ensucian la vista */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ESTILO DE LA TARJETA BLANCA (El contenedor principal) */
        /* Esto convierte el bloque central de Streamlit en la "tarjeta" */
        .block-container {
            background-color: white;
            padding: 3rem 3rem 4rem 3rem !important;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.08);
            max-width: 750px;
            margin-top: 2rem;
        }

        /* Títulos y textos */
        .badge {
            background-color: #2c3e50;
            color: white;
            padding: 6px 12px;
            border-radius: 50px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
            display: inline-block;
            margin-bottom: 10px;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 3.5rem !important;
            font-weight: 900 !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            line-height: 1 !important;
        }
        
        .highlight {
            color: #ef7d86;
        }

        h3 {
            color: #555;
            font-size: 14px !important;
            font-weight: 700 !important;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 0 !important;
        }

        .subtitle {
            font-size: 1.2rem;
            font-weight: 600;
            color: #34495e;
            margin-top: 10px;
            margin-bottom: 20px;
        }

        .description {
            color: #666;
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 25px;
        }

        /* Caja de Advertencia Personalizada */
        .warning-box {
            background-color: #f8f9fa;
            border-left: 5px solid #ef7d86;
            padding: 15px;
            border-radius: 5px;
            font-size: 0.85rem;
            color: #555;
            text-align: left;
            margin-bottom: 30px;
        }

        /* Personalización del botón nativo de Streamlit */
        .stButton > button {
            background: linear-gradient(90deg, #ef7d86 0%, #e8aeb3 100%);
            color: white !important;
            border: none;
            padding: 12px 35px;
            border-radius: 50px;
            font-weight: bold;
            font-size: 16px;
            box-shadow: 0 5px 15px rgba(239, 125, 134, 0.4);
            transition: all 0.3s ease;
            width: 100%;
        }

        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 20px rgba(239, 125, 134, 0.6);
            border-color: transparent !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# 4. Lógica de Pantallas
if st.session_state.page == 'landing':
    # --- PANTALLA DE INICIO (PORTADA) ---
    
    # Usamos columnas vacías para centrar textos si es necesario, 
    # pero aquí usaremos HTML directo para el encabezado para mayor control
    st.markdown("""
        <div style="text-align: center;">
            <div class="badge">TFM • Máster en IA aplicada a la salud</div>
            <h3>Centro Europeo de Másteres y Posgrados</h3>
            <h1>DIABETES<span class="highlight">.NME</span></h1>
            <div class="subtitle">Prototipo de CDSS para el diagnóstico temprano de diabetes</div>
            <p class="description">
                Este proyecto explora el potencial de integrar modelos predictivos avanzados en el flujo de trabajo 
                clínico, visualizando un futuro donde la IA actúa como un potente aliado en la detección temprana y 
                prevención de la diabetes tipo 2.
            </p>
            <div class="warning-box">
                <b>⚠️ Nota Académica:</b> Esta herramienta NO es un dispositivo médico certificado. 
                Los resultados son una simulación académica y NO deben utilizarse para el diagnóstico real, 
                tratamiento o toma de decisiones clínicas.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Botón de acción (Centrado usando columnas de Streamlit)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("INICIAR SIMULACIÓN ➔"):
            iniciar_simulacion()
            st.rerun()

elif st.session_state.page == 'simulacion':
    # --- PANTALLA DE LA APLICACIÓN (SIMULACIÓN) ---
    
    st.button("⬅ Volver al inicio", on_click=volver_inicio)
    
    st.title("Panel de Diagnóstico")
    st.write("Aquí irían tus inputs de datos del paciente, formulario y el modelo de IA.")
    
    # Ejemplo de estructura para tu TFM
    with st.expander("Datos del Paciente", expanded=True):
        c1, c2 = st.columns(2)
        c1.number_input("Edad", 18, 100, 45)
        c2.number_input("Glucosa", 50, 300, 100)
        c1.number_input("IMC", 15.0, 50.0, 24.5)
        c2.selectbox("Antecedentes Familiares", ["No", "Sí"])
    
    if st.button("Ejecutar Predicción"):
        st.success("El modelo ha procesado los datos correctamente.")
