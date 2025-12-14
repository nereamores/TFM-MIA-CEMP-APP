import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="CEMP Diabetes Predictor",
    page_icon="🏥",
    layout="wide"
)

# --- COLORES DE MARCA ---
COLOR_TITULOS = "#676A73"  # Gris elegante
COLOR_CEMP = "#E97F87"     # Rosita corporativo (Highlight/Riesgo)
COLOR_FONDO_NEUTRO = "#F0F2F6"

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown(f"""
    <style>
    /* Importar fuente (opcional, ejemplo Roboto) */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Roboto', sans-serif;
        color: {COLOR_TITULOS};
    }}
    
    /* Color de los títulos */
    h1, h2, h3 {{
        color: {COLOR_TITULOS} !important;
    }}
    
    /* Estilo del botón principal */
    div.stButton > button {{
        background-color: {COLOR_CEMP};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }}
    div.stButton > button:hover {{
        background-color: #D66A72; /* Un poco más oscuro al pasar el mouse */
        color: white;
    }}
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background-color: {COLOR_CEMP} !important;
        color: white !important;
    }}
    .stTabs [data-baseweb="tab-list"] button {{
        color: {COLOR_TITULOS};
    }}
    
    /* Cajas de métricas en sidebar */
    div[data-testid="stMetricValue"] {{
        color: {COLOR_CEMP} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- SIMULACIÓN DEL MODELO (Placeholder) ---
# En producción, aquí cargarías: model = pickle.load(open('modelo.pkl', 'rb'))
class MockModel:
    def predict_proba(self, X):
        # Lógica falsa para demo: Glucosa y BMI pesan mucho
        glucose = X[0]
        bmi = X[1]
        score = (glucose * 0.6) + (bmi * 0.4)
        prob = 1 / (1 + np.exp(-(score - 100) / 15)) # Sigmoide simple
        return [[1-prob, prob]]

model = MockModel()

# --- BARRA LATERAL (INPUTS) ---
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/E97F87/FFFFFF?text=CEMP+Health", use_container_width=True) # Logo Placeholder
    st.markdown("### Datos del Paciente")
    
    # Inputs solicitados
    pregnancies = st.number_input("Embarazos", 0, 20, 1)
    glucose = st.number_input("Glucosa (mg/dL)", 0, 300, 120)
    blood_pressure = st.number_input("Presión Arterial (mm Hg)", 0, 200, 70) # Añadido por completitud
    skin_thickness = st.number_input("Grosor Piel (mm)", 0, 100, 20) # Añadido por completitud
    insulin = st.number_input("Insulina (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 28.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Edad", 0, 120, 35)
    
    # --- CÁLCULOS AUTOMÁTICOS ---
    resistance_index = glucose * insulin
    bmi_squared = bmi ** 2
    is_prediabetes = "SÍ" if glucose > 140 else "NO"
    
    st.markdown("---")
    st.markdown("### 🧬 Variables Autocalculadas")
    
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        st.metric("Resistencia", f"{resistance_index:.0f}")
    with col_calc2:
        st.metric("BMI²", f"{bmi_squared:.0f}")
        
    # Alerta visual de Prediabetes
    if is_prediabetes == "SÍ":
        st.markdown(f"""
            <div style="background-color: {COLOR_CEMP}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">
                ⚠️ Marcador Prediabetes
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color: {COLOR_TITULOS}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">
                ✅ Sin Prediabetes
            </div>
        """, unsafe_allow_html=True)

# --- PREPARAR DATOS PARA EL MODELO ---
input_data = [glucose, bmi, insulin, age, pregnancies, dpf] # Orden según tu modelo real

# Predicción
probability = model.predict_proba(input_data)[0][1]
threshold = 0.27
prediction_class = 1 if probability > threshold else 0

# --- TÍTULO PRINCIPAL ---
st.title("Sistema de Predicción de Diabetes Mellitus")
st.markdown("Herramienta de soporte a la decisión clínica basada en Machine Learning.")

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(["🏥 Diagnóstico Clínico", "🔍 Explicabilidad (SHAP)", "🎛️ Simulación Terapéutica"])

# --- TAB 1: DIAGNÓSTICO ---
with tab1:
    col_res1, col_res2 = st.columns([2, 1])
    
    with col_res1:
        st.subheader("Resultado del Análisis")
        if prediction_class == 1:
            st.markdown(f"""
                <div style="background-color: {COLOR_CEMP}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h2 style="color: white !important; margin:0;">ALTO RIESGO DETECTADO</h2>
                    <p style="margin:0; font-size: 1.2rem;">Se recomienda intervención clínica inmediata.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: {COLOR_TITULOS}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h2 style="color: white !important; margin:0;">BAJO RIESGO</h2>
                    <p style="margin:0; font-size: 1.2rem;">Mantener hábitos saludables y control rutinario.</p>
                </div>
            """, unsafe_allow_html=True)
            
        # Contexto Clínico
        st.info("ℹ️ **Nota Clínica (ADA Standards):** Pacientes con glucosa en ayunas > 126 mg/dL o HbA1c > 6.5% deben ser confirmados con una segunda prueba.")

    with col_res2:
        st.subheader("Probabilidad")
        # Gráfico de gauge simple con CSS/HTML
        st.markdown(f"""
            <div style="text-align: center;">
                <h1 style="font-size: 4rem; color: {COLOR_CEMP if prediction_class == 1 else COLOR_TITULOS}; margin: 0;">
                    {probability*100:.1f}%
                </h1>
                <p>Certeza del Modelo</p>
                <small style="color: grey;">Umbral de decisión: {threshold}</small>
            </div>
        """, unsafe_allow_html=True)

# --- TAB 2: EXPLICABILIDAD (SHAP) ---
with tab2:
    st.subheader("Análisis de Factores de Riesgo (SHAP)")
    st.write("Este gráfico desglosa qué variables específicas empujaron la predicción hacia el riesgo (Rosita) o hacia la salud (Gris).")
    
    # --- GENERADOR DE GRÁFICO SHAP MANUAL (Para control total de colores) ---
    # En producción usarías: shap_values = explainer(X)
    # Aquí simulamos los valores para que el diseño sea exacto
    
    feature_names = ["Glucosa", "BMI", "Edad", "Insulina", "DPF", "Embarazos"]
    # Valores simulados: Positivos (aumentan riesgo), Negativos (bajan riesgo)
    shap_vals = [0.15, 0.10, -0.05, 0.02, -0.03, 0.01] 
    base_value = 0.3
    
    # Crear gráfico waterfall personalizado
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Posiciones
    y_pos = np.arange(len(feature_names))
    current_value = base_value
    
    for i, (name, val) in enumerate(zip(feature_names, shap_vals)):
        color = COLOR_CEMP if val > 0 else COLOR_TITULOS
        # Barra
        ax.barh(i, val, left=current_value, color=color, edgecolor='white', height=0.6)
        # Línea conectora
        if i < len(feature_names) - 1:
            ax.plot([current_value + val, current_value + val], [i, i-1], color="silver", linestyle="--", linewidth=0.5)
        
        # Etiquetas de texto
        text_x = current_value + val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax.text(text_x, i, f"{val:+.2f}", va='center', ha=ha, fontsize=9, color=COLOR_TITULOS, fontweight='bold')
        
        current_value += val
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=11, color=COLOR_TITULOS)
    ax.set_xlabel("Impacto en la Probabilidad", color=COLOR_TITULOS)
    ax.set_title(f"Interpretación para el Paciente Actual (Base: {base_value})", color=COLOR_TITULOS)
    
    # Quitar bordes feos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_TITULOS)
    ax.spines['bottom'].set_color(COLOR_TITULOS)
    ax.tick_params(axis='x', colors=COLOR_TITULOS)
    
    st.pyplot(fig)

# --- TAB 3: SIMULACIÓN ---
with tab3:
    st.subheader("Simulador Terapéutico 'What-If'")
    st.markdown("Ajuste los valores objetivo para visualizar la reducción potencial del riesgo.")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.markdown("#### 🎯 Objetivos Terapéuticos")
        new_glucose = st.slider("Nueva Glucosa Meta", 50, 200, glucose)
        new_bmi = st.slider("Nuevo BMI Meta", 15.0, 50.0, bmi)
        
        # Recalcular riesgo simulado
        sim_input = [new_glucose, new_bmi, insulin, age, pregnancies, dpf]
        sim_prob = model.predict_proba(sim_input)[0][1]
        reduction = probability - sim_prob
        
    with col_sim2:
        st.markdown("#### 📊 Comparativa de Riesgo")
        
        # Gráfico de barras comparativo simple
        fig_sim, ax_sim = plt.subplots(figsize=(6, 4))
        categories = ['Actual', 'Simulado']
        values = [probability, sim_prob]
        colors = [COLOR_CEMP, '#D3D3D3'] # Rosita vs Gris claro
        
        bars = ax_sim.bar(categories, values, color=colors, width=0.5)
        ax_sim.set_ylim(0, 1)
        ax_sim.set_ylabel("Probabilidad de Diabetes")
        
        # Etiquetas encima de las barras
        for bar in bars:
            height = bar.get_height()
            ax_sim.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%',
                    ha='center', va='bottom', fontweight='bold', color=COLOR_TITULOS)
            
        # Quitar bordes
        ax_sim.spines['top'].set_visible(False)
        ax_sim.spines['right'].set_visible(False)
        
        st.pyplot(fig_sim)
        
        if reduction > 0:
            st.success(f"📉 ¡Excelente! Con estos cambios, el riesgo se reduciría un **{reduction*100:.1f}%**.")
        else:
            st.info("Ajuste los valores para ver mejoras.")
