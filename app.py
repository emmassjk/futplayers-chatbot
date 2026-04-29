"""
app.py
======
Interfaz Streamlit del chatbot FutPlayers.cl
Ejecución: streamlit run app.py
"""

import sys
import importlib.util
from pathlib import Path
import streamlit as st


def _importar(nombre_archivo: str, alias: str):
    ruta = Path(__file__).parent / "src" / nombre_archivo
    spec = importlib.util.spec_from_file_location(alias, ruta)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

_m1 = _importar("1_conexion_llm.py",  "m_conexion")
_m2 = _importar("2_prompts.py",       "m_prompts")
_m3 = _importar("3_rag_pipeline.py",  "m_rag")
_m5 = _importar("5_evaluacion.py",    "m_eval")
_m4 = _importar("4_agente.py",        "m_agente")

FutPlayersAgent = _m4.FutPlayersAgent
evaluar_sistema = _m5.evaluar_sistema

st.set_page_config(page_title="FutPlayers.cl — Chatbot", page_icon="⚽", layout="centered")

st.markdown("""
<style>
    .titulo-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="titulo-header">
    <h2>⚽ FutPlayers.cl</h2>
    <p style="margin:0; opacity:0.85;">Asistente virtual de ventas — Poleras de fútbol oficiales</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Panel de Control")

    if st.button("🔄 Nueva Conversación", use_container_width=True):
        if "agente" in st.session_state:
            st.session_state.agente.limpiar_memoria()
        st.session_state.mensajes = []
        st.rerun()

    st.divider()
    st.markdown("### 📋 Preguntas de Ejemplo")
    ejemplos = [
        "¿Qué equipos tienen disponibles?",
        "¿Tienen polera del Real Madrid talla L?",
        "¿Cuánto cuesta el envío a Concepción?",
        "¿Hay descuento por comprar 2 poleras?",
        "¿Puedo pagar con transferencia?",
        "¿Puedo cambiar la talla si no me queda?",
    ]
    for ej in ejemplos:
        if st.button(ej, use_container_width=True, key=f"ej_{ej[:20]}"):
            st.session_state.pregunta_rapida = ej

    st.divider()
    st.markdown("### 📊 Evaluación RAG")
    if st.button("▶ Ejecutar Evaluación", use_container_width=True):
        st.session_state.mostrar_evaluacion = True

    st.divider()
    st.caption("ISY0101 · Ingeniería de Soluciones con IA · DuocUC")


@st.cache_resource(show_spinner="🚀 Cargando el chatbot...")
def obtener_agente():
    return FutPlayersAgent()


if "agente" not in st.session_state:
    st.session_state.agente = obtener_agente()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {
            "rol": "bot",
            "texto": "¡Hola! 👋 Soy **Futbot**, tu asesor de FutPlayers.cl ⚽\n\n"
                     "Puedo ayudarte con:\n"
                     "- 👕 Consultar stock y precios de poleras\n"
                     "- 🚚 Información de envíos\n"
                     "- 💳 Medios de pago\n"
                     "- 🔄 Cambios y devoluciones\n\n"
                     "¿En qué te puedo ayudar hoy?",
        }
    ]

for msg in st.session_state.mensajes:
    avatar = "⚽" if msg["rol"] == "bot" else "👤"
    with st.chat_message(msg["rol"], avatar=avatar):
        st.markdown(msg["texto"])

if "pregunta_rapida" in st.session_state:
    pregunta_sidebar = st.session_state.pop("pregunta_rapida")
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta_sidebar})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta_sidebar)
    with st.chat_message("assistant", avatar="⚽"):
        placeholder = st.empty()
        respuesta_acum = ""
        for token in st.session_state.agente.responder_streaming(pregunta_sidebar):
            respuesta_acum += token
            placeholder.markdown(respuesta_acum + "▌")
        placeholder.markdown(respuesta_acum)
    st.session_state.mensajes.append({"rol": "bot", "texto": respuesta_acum})
    st.rerun()

if pregunta := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)
    with st.chat_message("assistant", avatar="⚽"):
        placeholder = st.empty()
        respuesta_acum = ""
        for token in st.session_state.agente.responder_streaming(pregunta):
            respuesta_acum += token
            placeholder.markdown(respuesta_acum + "▌")
        placeholder.markdown(respuesta_acum)
    st.session_state.mensajes.append({"rol": "bot", "texto": respuesta_acum})

if st.session_state.get("mostrar_evaluacion", False):
    st.divider()
    st.markdown("## 📊 Evaluación del Sistema RAG")
    with st.spinner("Evaluando..."):
        resultados = evaluar_sistema(st.session_state.agente)
    prom = resultados["promedios"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Context Precision", f"{prom['context_precision_avg']:.0%}")
    col2.metric("Context Recall",    f"{prom['context_recall_avg']:.0%}")
    col3.metric("Faithfulness",      f"{prom['faithfulness_avg']:.0%}")
    col4.metric("Answer Relevancy",  f"{prom['answer_relevancy_avg']:.0%}")
    st.markdown("### Detalle por Pregunta")
    for r in resultados["detalle"]:
        with st.expander(f"❓ {r['pregunta']}"):
            st.write(f"**Respuesta:** {r['respuesta']}")
            st.write(f"**Context Precision:** {r['context_precision']:.2f}")
            st.write(f"**Context Recall:** {r['context_recall']:.2f}")
            st.write(f"**Faithfulness:** {r['faithfulness']:.2f} — {r['faithfulness_razon']}")
            st.write(f"**Answer Relevancy:** {r['answer_relevancy']:.2f} — {r['relevancia_razon']}")
    st.session_state.mostrar_evaluacion = False