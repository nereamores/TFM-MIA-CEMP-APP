import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import datetime

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="DIABETES.NME",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# MODELO SIMULADO
# =========================================================
class MockModel:
    def predict_proba(self, X):
        score = (X[0]*0.5) + (X[1]*0.4) + (X[3]*0.1)
        prob = 1 / (1 + np.exp(-(score - 100) / 15))
        return [[1 - prob, prob]]

if "model" not in st.session_state:
    st.session_state.model = MockModel()

if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if "page" not in st.session_state:
    st.session_state.page = "landing"

# =========================================================
# UTILIDADES
# =========================================================
def fig_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=300)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img}" style="width:100%;">'

def help_icon(txt):
    return f"""<span title="{txt}" style="display:inline-block;width:16px;height:16px;
    line-height:16px;text-align:center;border-radius:50%;background:#E0E0E0;
    color:#777;font-size:0.7rem;font-weight:bold;margin-left:6px;">?</span>"""

# =========================================================
# PORTADA
# =========================================================
if st.session_state.page == "landing":

    st.markdown("""
    <style>
    #MainMenu, footer, header {visibility:hidden;}
    .block-container {
        max-width:800px;
        margin:auto;
        background:white;
        padding:3rem;
        border-radius:20px;
        box-shadow:0 10px 30px rgba(0,0,0,.05);
    }
    h1 {text-align:center;font-size:3.5rem;font-weight:800;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h1>D<span style="color:#ef7d86">IA</span>BETES<span style="color:#bdc3c7">.</span><span style="color:#ef7d86">NME</span></h1>
    <p style="text-align:center;font-weight:700;">
    Prototipo CDSS para diagnóstico temprano de diabetes
    </p>
    <p style="text-align:center;color:#666;">
    Aplicación académica – NO es dispositivo médico
    </p>
    """, unsafe_allow_html=True)

    if st.button("INICIAR"):
        st.session_state.page = "simulacion"
        st.rerun()

# =========================================================
# SIMULACIÓN
# =========================================================
else:

    CEMP_PINK = "#E97F87"
    CEMP_DARK = "#2C3E50"
    GOOD_TEAL = "#4DB6AC"

    st.markdown("""
    <style>
    #MainMenu, footer {visibility:hidden;}
    .card {
        background:white;
        border-radius:12px;
        padding:20px;
        box-shadow:0 4px 15px rgba(0,0,0,.04);
        margin-bottom:15px;
    }
    .card-header {
        font-size:.75rem;
        letter-spacing:1px;
        font-weight:700;
        color:#999;
        margin-bottom:10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        if st.button("⬅ Volver"):
            st.session_state.page = "landing"
            st.rerun()

        patient_name = st.text_input("ID Paciente", "Paciente #8842-X")
        date = st.date_input("Fecha", datetime.date.today())

        glucose = st.slider("Glucosa 2h", 50, 350, 120)
        insulin = st.slider("Insulina", 0, 900, 100)
        weight = st.slider("Peso (kg)", 30.0, 200.0, 70.0)
        height = st.slider("Altura (m)", 1.0, 2.2, 1.7)
        age = st.slider("Edad", 18, 90, 45)

    bmi = weight / (height ** 2)

    input_data = [glucose, bmi, insulin, age, 0, 0]
    prob = st.session_state.model.predict_proba(input_data)[0][1]

    threshold = 0.27
    is_high = prob > threshold

    risk_label = "ALTO RIESGO" if is_high else "BAJO RIESGO"
    risk_color = CEMP_PINK if is_high else GOOD_TEAL
    risk_bg = "#FFF5F5" if is_high else "#F0FDF4"
    risk_icon = "🔴" if is_high else "🟢"

    # =======================
    # FICHA PACIENTE (FIX)
    # =======================
    badges_html = f"""
    <div style="background:{risk_bg};border:1px solid {risk_color};
    color:{risk_color};font-weight:800;padding:8px 16px;border-radius:30px;">
    {risk_icon} {risk_label}
    </div>
    """

    st.markdown(f"""
    <div class="card" style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <span class="card-header">EXPEDIENTE MÉDICO</span>
            <h2 style="margin:0;color:{CEMP_DARK};">{patient_name}</h2>
            <div style="color:#666;font-size:.85rem;">📅 {date}</div>
        </div>
        {badges_html}
    </div>
    """, unsafe_allow_html=True)

    # =======================
    # HALLAZGOS
    # =======================
    if glucose >= 200:
        txt = "Posible Diabetes"
        icon = "⚠️"
        color = CEMP_PINK
    else:
        txt = "Sin hallazgos significativos"
        icon = "✅"
        color = GOOD_TEAL

    st.markdown(f"""
    <div class="card" style="border-left:5px solid {color};">
        <span class="card-header" style="color:{color};">HALLAZGOS CLAVE</span>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <h3 style="margin:0;">{txt}</h3>
            <div style="font-size:1.8rem;">{icon}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("CALCULAR RIESGO", use_container_width=True):
        st.session_state.predict_clicked = True
        st.rerun()

    # =======================
    # DONUT
    # =======================
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    ax.pie(
        [prob, 1 - prob],
        colors=[risk_color, "#F4F6F9"],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.2)
    )

    center = f"{prob*100:.1f}%"

    html_img = fig_to_html(fig)
    plt.close(fig)

    st.markdown(f"""
    <div class="card" style="text-align:center;">
        <span class="card-header">PROBABILIDAD IA {help_icon("Salida del modelo")}</span>
        <div style="position:relative;">
            {html_img}
            <div style="position:absolute;top:50%;left:50%;
            transform:translate(-50%,-50%);
            font-size:2.3rem;font-weight:800;color:{CEMP_DARK};">
            {center}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
