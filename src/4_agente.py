"""
src/4_agente.py
===============
RA2/IL2.1 + RA2/IL2.2 - Agente conversacional con memoria + RAG
"""

import os
import importlib
import sys as _sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Importar módulos propios con importlib
def _importar_modulo(nombre_archivo: str, alias: str):
    ruta = Path(__file__).parent / nombre_archivo
    spec = importlib.util.spec_from_file_location(alias, ruta)
    mod  = importlib.util.module_from_spec(spec)
    _sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

_m1 = _importar_modulo("1_conexion_llm.py",  "src_1_conexion_llm")
_m2 = _importar_modulo("2_prompts.py",        "src_2_prompts")
_m3 = _importar_modulo("3_rag_pipeline.py",   "src_3_rag_pipeline")

crear_llm             = _m1.crear_llm
crear_prompt_template = _m2.crear_prompt_template
crear_prompt_cot      = _m2.crear_prompt_cot
obtener_retriever     = _m3.obtener_retriever

PALABRAS_COT = ["descuento", "cuánto pagaría", "total", "precio final",
                "cuántas", "comparar", "conviene", "calcul"]

def requiere_cot(pregunta: str) -> bool:
    return any(p in pregunta.lower() for p in PALABRAS_COT)


class FutPlayersAgent:
    """
    Agente conversacional de ventas para FutPlayers.cl.
    Arquitectura: Usuario → RAG → Prompt + Contexto → LLM → Respuesta
    Memoria: lista de HumanMessage/AIMessage (ConversationBuffer equivalente)
    """

    def __init__(self):
        print("🤖 Inicializando FutPlayersAgent...")
        self.llm           = crear_llm(temperature=0.3, streaming=True)
        self.retriever     = obtener_retriever()
        self.prompt_normal = crear_prompt_template()
        self.prompt_cot    = crear_prompt_cot()
        self.chat_history: list = []
        print("✅ Agente listo.\n")

    def _recuperar_contexto(self, pregunta: str) -> str:
        docs = self.retriever.invoke(pregunta)
        if not docs:
            return "No se encontró información específica en el catálogo."
        partes = [f"[Fuente: {d.metadata.get('fuente','?')}]\n{d.page_content}" for d in docs]
        return "\n\n".join(partes)

    def responder(self, pregunta: str) -> str:
        contexto = self._recuperar_contexto(pregunta)
        prompt   = self.prompt_cot if requiere_cot(pregunta) else self.prompt_normal
        chain    = prompt | self.llm

        respuesta = chain.invoke({
            "context":      contexto,
            "chat_history": self.chat_history,
            "input":        pregunta,
        })

        self.chat_history.append(HumanMessage(content=pregunta))
        self.chat_history.append(AIMessage(content=respuesta.content))
        return respuesta.content

    def responder_streaming(self, pregunta: str):
        contexto = self._recuperar_contexto(pregunta)
        prompt   = self.prompt_cot if requiere_cot(pregunta) else self.prompt_normal
        chain    = prompt | crear_llm(temperature=0.3, streaming=True)

        respuesta_completa = ""
        for chunk in chain.stream({
            "context":      contexto,
            "chat_history": self.chat_history,
            "input":        pregunta,
        }):
            token = chunk.content
            respuesta_completa += token
            yield token

        self.chat_history.append(HumanMessage(content=pregunta))
        self.chat_history.append(AIMessage(content=respuesta_completa))

    def limpiar_memoria(self):
        self.chat_history = []
        print("🔄 Memoria de sesión limpiada.")

    def get_historial(self) -> list[dict]:
        resultado = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                resultado.append({"rol": "usuario", "texto": msg.content})
            elif isinstance(msg, AIMessage):
                resultado.append({"rol": "bot", "texto": msg.content})
        return resultado


if __name__ == "__main__":
    agente = FutPlayersAgent()
    preguntas = [
        "Hola, ¿qué equipos tienen disponibles?",
        "¿Tienen polera del Liverpool en talla M?",
        "¿Cuánto cuesta el envío a Temuco?",
        "Si compro 3 poleras, ¿me hacen descuento?",
        "¿Puedo pagar con transferencia bancaria?",
    ]
    print("=" * 60)
    print("🤖 CHATBOT FUTPLAYERS.CL - PRUEBA EN CONSOLA")
    print("=" * 60)
    for pregunta in preguntas:
        print(f"\n👤 Usuario: {pregunta}")
        print(f"🤖 Futbot:  {agente.responder(pregunta)}")
        print("-" * 40)