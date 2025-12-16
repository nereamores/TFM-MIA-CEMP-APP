import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import date
import time

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="DIABETES.NME | TFM",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. GESTIÓN DE ESTADO ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'patient' not in st.session_state: st.session_state.patient = {'id': '', 'name': '', 'date': date.today()}
if 'threshold' not in st.session_state: st.session_state.threshold = 0.27
if 'model' not in st.session_state:
    class MockModel:
        def predict_proba(self, X):
            score = (X[0]*0.5) + (X[1]*0.6) + (X[3]*0.1)
            prob = 1 / (1 + np.exp(-(score - 110) / 20))
            return [[1-prob, prob]]
    st.session_state.model = MockModel()

# --- 3. COLORES ---
CEMP_PINK = "#E97F87"
CEMP_DARK = "#2C3E50"
SLIDER_GRAY = "#BDC3C7"
GOOD_TEAL = "#4DB6AC"

# --- 4. ESTILOS ---
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

#MainMenu, footer, header {{
    visibility: hidden;
}}

.block-container {{
    max-width: 1250px;
    padding-top: 1.5rem;
}}

.landing-wrapper {{
    background: linear-gradient(145deg, #FFFFFF 0%, #FFF5F6 100%);
    padding: 50px 40px;
    border-radius: 20px;
    border: 1px solid rgba(233,127,135,0.15);
    box-shadow: 0 20px 40px rgba(233,127,135,0.08);
    max-width: 900px;
    margin: 30px auto;
    text-align: center;
}}

.cemp-badge {{
    display: inline-block;
    background-color: {CEMP_DARK};
    color: white;
    padding: 6px 16px;
    border-radius: 30px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 20px;
}}

.landing-institution {{
    font-weight: 700;
    font-size: 1rem;
    color: {CEMP_DARK};
    text-transform: uppercase;
    letter-spacing: 1.2px;
}}

.landing-title-text {{
    font-weight: 900;
    font-size: 3.5rem;
    color: {CEMP_DARK};
    margin: 10px 0 20px;
}}

.landing-pink {{ color: {CEMP_PINK}; }}
.landing-gray {{ color: {SLIDER_GRAY}; }}

.landing-hero-text {{
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 20px;
    color: {CEMP_DARK};
}}

.landing-content {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 25px;
}}

.landing-description {{
    max-width: 700px;
    text-align: justify;
    font-size: 1rem;
    line-height: 1.6;
    color: #666;
}}

.disclaimer-box {{
    background: #F8F9FA;
    border-left: 4px solid {CEMP_PINK};
    padding: 20px;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #555;
    max-width: 750px;
}}

div.stButton > button {{
    background-color: {CEMP_PINK};
    color: white;
    font-weight: 800;
    font-size: 1.1rem;
    padding: 1rem 3rem;
    border-radius: 50px;
    border: none;
    letter-spacing: 1px;
    box-shadow: 0 10px 25px rgba(233,127,135,0.4);
    transition: all 0.3s ease;
}}

div.stButton > button:hover {{
    background-color: #D66E76;
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(233,127,135,0.5);
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# PASO 1 — PORTADA
# =====================================================
if st.session_state.step == 1:

    st.markdown("""
    <div class="landing-wrapper">

        <div class="cemp-badge">
            TFM • MÁSTER EN INTELIGENCIA ARTIFICIAL APLICADA A LA SALUD
        </div>

        <div class="landing-institution">
            CENTRO EUROPEO DE MÁSTERES Y POSGRADOS
        </div>

        <div class="landing-title-text">
            D<span class="landing-pink">IA</span>BETES<span class="landing-gray">.</span><span class="landing-pink">NME</span>
        </div>

        <div class="landing-hero-text">
            Prototipo de CDSS para el diagnóstico temprano de diabetes
        </div>

        <div class="landing-content">

            <p class="landing-description">
                Este proyecto explora el potencial de integrar modelos predictivos avanzados
                en el flujo de trabajo clínico, visualizando un futuro donde la IA actúa como
                un potente aliado en la detección temprana y prevención de la diabetes tipo 2.
            </p>

            <div class="disclaimer-box">
                <strong>Aplicación desarrollada con fines exclusivamente educativos como parte de un Trabajo de Fin de Máster.</strong><br><br>
                ⚠️ Esta herramienta NO es un dispositivo médico certificado.  
                Los resultados son una simulación académica y NO deben utilizarse para el diagnóstico real,
                tratamiento o toma de decisiones clínicas.
            </div>

        </div>
    </div>
    """, unsafe_allow_html=True)

    # BOTÓN DENTRO DEL RECTÁNGULO
    st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
    if st.button("INICIAR SIMULACIÓN  ➔"):
        st.session_state.step = 2
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PASO 2 Y SIGUIENTES (tu código continúa igual)
# =====================================================
