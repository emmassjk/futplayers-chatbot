"""
src/1_conexion_llm.py
=====================
IL1.1 - Conexión al LLM mediante LangChain + GitHub Models API
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Cargar .env desde la raíz del proyecto
env_path = Path(__file__).resolve().parent.parent / ".env"

print("=" * 50)
print("Directorio actual:", Path.cwd())
print("Ruta .env:", env_path)
print("Existe .env:", env_path.exists())
print("=" * 50)

load_dotenv(dotenv_path=env_path)

print("GITHUB_TOKEN:", os.getenv("GITHUB_TOKEN"))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))
print("LLM_MODEL:", os.getenv("LLM_MODEL"))
print("=" * 50)

token = os.getenv("GITHUB_TOKEN")

if not token:
    raise Exception(
        f"GITHUB_TOKEN no encontrado. Valor actual: {token}"
    )


def crear_llm(temperature: float = 0.3, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://models.inference.ai.azure.com"
        ),
        api_key=token,
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        streaming=streaming,
    )


def probar_conexion():
    print("Probando conexión con el LLM...")

    llm = crear_llm()

    messages = [
        SystemMessage(
            content="Eres un asistente de prueba. Responde muy brevemente."
        ),
        HumanMessage(
            content="¿Estás disponible? Responde solo 'Sí, conexión exitosa.'"
        ),
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