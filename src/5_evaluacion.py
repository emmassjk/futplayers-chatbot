"""
src/5_evaluacion.py
===================
IL1.4 - Evaluación del pipeline RAG
Métricas: Context Precision, Context Recall, Faithfulness, Answer Relevancy
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DATASET_EVALUACION = [
    {
        "pregunta": "¿Tienen polera del Real Madrid talla L?",
        "palabras_clave": ["real madrid", "talla l", "29990", "stock"],
    },
    {
        "pregunta": "¿Cuánto cuesta el envío a Valparaíso?",
        "palabras_clave": ["3990", "39990", "valparaíso", "envío"],
    },
    {
        "pregunta": "¿Aceptan transferencia bancaria?",
        "palabras_clave": ["transferencia", "banco", "24 horas"],
    },
    {
        "pregunta": "¿Qué descuento hay si compro 2 poleras del mismo equipo?",
        "palabras_clave": ["10%", "10 por ciento", "descuento", "segunda"],
    },
    {
        "pregunta": "¿Puedo devolver una polera que no me quedó bien?",
        "palabras_clave": ["10 días", "devolución", "etiquetas", "sin uso"],
    },
]


def crear_llm_juez() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("GITHUB_TOKEN"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


PROMPT_FIDELIDAD = ChatPromptTemplate.from_template("""
Evalúa si la respuesta del chatbot se basa SOLO en el contexto provisto.
Responde únicamente con un JSON: {{"score": 0.0_a_1.0, "razon": "explicación breve"}}

Contexto utilizado:
{contexto}

Respuesta del chatbot:
{respuesta}

Criterio: score=1.0 si toda la info proviene del contexto. score=0.0 si inventa información.
""")

PROMPT_RELEVANCIA = ChatPromptTemplate.from_template("""
Evalúa si la respuesta responde directamente la pregunta del usuario.
Responde únicamente con un JSON: {{"score": 0.0_a_1.0, "razon": "explicación breve"}}

Pregunta: {pregunta}
Respuesta: {respuesta}

Criterio: score=1.0 si es directa y útil. score=0.0 si evade o no responde.
""")


def calcular_context_precision(docs: list, palabras_clave: list) -> float:
    if not docs:
        return 0.0
    relevantes = sum(
        1 for doc in docs
        if any(kw.lower() in doc.page_content.lower() for kw in palabras_clave)
    )
    return round(relevantes / len(docs), 2)


def calcular_context_recall(docs: list, palabras_clave: list) -> float:
    if not palabras_clave:
        return 0.0
    texto = " ".join(doc.page_content.lower() for doc in docs)
    cubiertas = sum(1 for kw in palabras_clave if kw.lower() in texto)
    return round(cubiertas / len(palabras_clave), 2)


def calcular_faithfulness(contexto: str, respuesta: str, llm) -> dict:
    try:
        resultado = (PROMPT_FIDELIDAD | llm).invoke({"contexto": contexto, "respuesta": respuesta})
        return json.loads(resultado.content)
    except Exception:
        return {"score": 0.5, "razon": "Error al evaluar"}


def calcular_relevancia(pregunta: str, respuesta: str, llm) -> dict:
    try:
        resultado = (PROMPT_RELEVANCIA | llm).invoke({"pregunta": pregunta, "respuesta": respuesta})
        return json.loads(resultado.content)
    except Exception:
        return {"score": 0.5, "razon": "Error al evaluar"}


def evaluar_sistema(agente) -> dict:
    print("\n📊 INICIANDO EVALUACIÓN DEL SISTEMA RAG")
    print("=" * 60)

    llm_juez   = crear_llm_juez()
    resultados = []

    for caso in DATASET_EVALUACION:
        pregunta       = caso["pregunta"]
        palabras_clave = caso["palabras_clave"]
        print(f"\n🔍 Evaluando: {pregunta[:50]}...")

        docs      = agente.retriever.invoke(pregunta)
        respuesta = agente.responder(pregunta)
        contexto  = "\n".join(d.page_content for d in docs)

        precision  = calcular_context_precision(docs, palabras_clave)
        recall     = calcular_context_recall(docs, palabras_clave)
        fidelidad  = calcular_faithfulness(contexto, respuesta, llm_juez)
        relevancia = calcular_relevancia(pregunta, respuesta, llm_juez)

        resultado = {
            "pregunta":           pregunta,
            "respuesta":          respuesta[:100] + "...",
            "context_precision":  precision,
            "context_recall":     recall,
            "faithfulness":       fidelidad["score"],
            "answer_relevancy":   relevancia["score"],
            "faithfulness_razon": fidelidad["razon"],
            "relevancia_razon":   relevancia["razon"],
        }
        resultados.append(resultado)

        print(f"  Context Precision: {precision:.2f}")
        print(f"  Context Recall:    {recall:.2f}")
        print(f"  Faithfulness:      {fidelidad['score']:.2f} — {fidelidad['razon']}")
        print(f"  Answer Relevancy:  {relevancia['score']:.2f} — {relevancia['razon']}")

    promedios = {
        "context_precision_avg": round(sum(r["context_precision"] for r in resultados) / len(resultados), 2),
        "context_recall_avg":    round(sum(r["context_recall"]    for r in resultados) / len(resultados), 2),
        "faithfulness_avg":      round(sum(r["faithfulness"]      for r in resultados) / len(resultados), 2),
        "answer_relevancy_avg":  round(sum(r["answer_relevancy"]  for r in resultados) / len(resultados), 2),
    }

    print("\n" + "=" * 60)
    print("📈 RESUMEN DE MÉTRICAS PROMEDIO:")
    for k, v in promedios.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    return {"promedios": promedios, "detalle": resultados}