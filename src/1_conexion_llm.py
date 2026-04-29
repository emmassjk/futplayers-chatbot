"""
src/1_conexion_llm.py
=====================
IL1.1 - Conexión al LLM mediante LangChain + GitHub Models API
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def crear_llm(temperature: float = 0.3, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com"),
        api_key=os.getenv("GITHUB_TOKEN"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        streaming=streaming,
    )


def probar_conexion():
    print("Probando conexión con el LLM...")
    llm = crear_llm()
    messages = [
        SystemMessage(content="Eres un asistente de prueba. Responde muy brevemente."),
        HumanMessage(content="¿Estás disponible? Responde solo 'Sí, conexión exitosa.'"),
    ]
    try:
        respuesta = llm.invoke(messages)
        print(f"✅ Conexión exitosa: {respuesta.content}")
        return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False


if __name__ == "__main__":
    probar_conexion()