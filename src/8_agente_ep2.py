"""
src/8_agente_ep2.py
===================
EP2 — IL2.1 + IL2.2 + IL2.3
Agente FutPlayers EP2: integración de herramientas, memoria y planificación.

Arquitectura completa:
                      ┌──────────────────────────────────────────┐
   Pregunta usuario   │            FutPlayersAgentEP2             │
         │            │                                          │
         ▼            │  ┌─────────────┐   ┌──────────────────┐  │
   ┌───────────┐      │  │ Planificador│   │  MemoryManager   │  │
   │  Entrada  │──────┼─▶│ (IL2.3)     │   │  (IL2.2)         │  │
   └───────────┘      │  │ Intent →    │   │  ShortTerm       │  │
                      │  │ TaskPlan    │   │  LongTerm        │  │
                      │  └──────┬──────┘   └────────┬─────────┘  │
                      │         │                   │             │
                      │         ▼                   ▼             │
                      │  ┌─────────────────────────────────────┐  │
                      │  │     SmartRetriever (RAG)            │  │
                      │  │     LangChain / Chroma              │  │
                      │  └──────────────┬──────────────────────┘  │
                      │                 │                          │
                      │                 ▼                          │
                      │  ┌─────────────────────────────────────┐  │
                      │  │     Prompt Engineering              │  │
                      │  │  system + contexto + historial      │  │
                      │  │  + prompt_hint del plan             │  │
                      │  └──────────────┬──────────────────────┘  │
                      │                 │                          │
                      │                 ▼                          │
                      │  ┌─────────────────────────────────────┐  │
                      │  │     LLM (GPT-4o-mini / streaming)   │  │
                      │  └──────────────┬──────────────────────┘  │
                      │                 │                          │
                      └─────────────────┼──────────────────────────┘
                                        │
                                        ▼
                                  Respuesta final

Cómo ejecutar prueba en consola:
    python src/8_agente_ep2.py
"""

from __future__ import annotations

import os
import importlib
import sys as _sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────
# IMPORTACIÓN DE MÓDULOS PROPIOS
# ─────────────────────────────────────────────────────────────

def _importar(nombre: str, alias: str):
    ruta = Path(__file__).parent / nombre
    spec = importlib.util.spec_from_file_location(alias, ruta)
    mod  = importlib.util.module_from_spec(spec)
    _sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

_m1  = _importar("1_conexion_llm.py",   "ep2_m1")
_m2  = _importar("2_prompts.py",        "ep2_m2")
_m3  = _importar("3_rag_pipeline.py",   "ep2_m3")
_m6  = _importar("6_memoria_larga.py",  "ep2_m6")
_m7  = _importar("7_planificacion.py",  "ep2_m7")

crear_llm             = _m1.crear_llm
crear_prompt_template = _m2.crear_prompt_template
crear_prompt_cot      = _m2.crear_prompt_cot
obtener_retriever     = _m3.obtener_retriever
MemoryManager         = _m6.MemoryManager
Planificador          = _m7.Planificador

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ─────────────────────────────────────────────────────────────
# AGENTE EP2
# ─────────────────────────────────────────────────────────────

class FutPlayersAgentEP2:
    """
    Agente conversacional EP2 para FutPlayers.cl.

    Mejoras respecto al agente EP1:
      ✅ Planificación basada en intención (IL2.3)
      ✅ Memoria corto plazo con ventana deslizante (IL2.2)
      ✅ Memoria largo plazo con preferencias semánticas (IL2.2)
      ✅ prompt_hint dinámico inyectado por el planificador (IL2.3)
      ✅ Contexto de preferencias enriquecido automáticamente (IL2.2)
      ✅ Herramientas RAG + LLM con selección adaptativa (IL2.1)
    """

    def __init__(self, ventana_memoria: int = 6):
        print("🤖 Inicializando FutPlayersAgentEP2...")
        self.llm           = crear_llm(temperature=0.3, streaming=True)
        self.retriever     = obtener_retriever()
        self.prompt_normal = crear_prompt_template()
        self.prompt_cot    = crear_prompt_cot()
        self.memoria       = MemoryManager(ventana_corto_plazo=ventana_memoria)
        self.planificador  = Planificador()
        print("✅ Agente EP2 listo.\n")

    # ── Herramienta 1: Recuperación de contexto ───────────────

    def _recuperar_contexto(self, pregunta: str) -> str:
        """Herramienta de consulta: invoca el SmartRetriever."""
        docs = self.retriever.invoke(pregunta)
        if not docs:
            return "No se encontró información específica en el catálogo."
        partes = [
            f"[Fuente: {d.metadata.get('fuente','?')}]\n{d.page_content}"
            for d in docs
        ]
        return "\n\n".join(partes)

    # ── Herramienta 2: Construcción de prompt adaptativo ─────

    def _construir_prompt(self, plan, contexto: str, preferencias: str) -> ChatPromptTemplate:
        """
        Herramienta de razonamiento: selecciona y enriquece el prompt
        según el plan generado por el planificador.
        """
        base_prompt = self.prompt_cot if plan.requiere_cot else self.prompt_normal

        if plan.prompt_hint or preferencias:
            # Inyectar hint del plan + preferencias en el contexto
            contexto_enriquecido = ""
            if preferencias:
                contexto_enriquecido += preferencias + "\n\n"
            if plan.prompt_hint:
                contexto_enriquecido += f"[Instrucción del planificador]\n{plan.prompt_hint}\n\n"
            contexto_enriquecido += contexto
            return base_prompt, contexto_enriquecido

        return base_prompt, contexto

    # ── Herramienta 3: Escritura de respuesta ────────────────

    def responder(self, pregunta: str) -> str:
        """Respuesta completa (no streaming) — útil para pruebas."""
        plan        = self.planificador.planificar(pregunta)
        contexto    = self._recuperar_contexto(pregunta) if plan.requiere_rag else ""
        preferencias = self.memoria.enriquecer_contexto()
        prompt, ctx = self._construir_prompt(plan, contexto, preferencias)
        chain       = prompt | self.llm

        respuesta = chain.invoke({
            "context":      ctx,
            "chat_history": self.memoria.obtener_historial_reciente(),
            "input":        pregunta,
        })

        self.memoria.agregar_turno(pregunta, respuesta.content)
        self._actualizar_preferencias_desde_plan(plan, pregunta)
        return respuesta.content

    def responder_streaming(self, pregunta: str):
        """Respuesta en streaming — compatible con Streamlit."""
        plan         = self.planificador.planificar(pregunta)
        contexto     = self._recuperar_contexto(pregunta) if plan.requiere_rag else ""
        preferencias = self.memoria.enriquecer_contexto()
        prompt, ctx  = self._construir_prompt(plan, contexto, preferencias)
        chain        = prompt | crear_llm(temperature=0.3, streaming=True)

        respuesta_completa = ""
        for chunk in chain.stream({
            "context":      ctx,
            "chat_history": self.memoria.obtener_historial_reciente(),
            "input":        pregunta,
        }):
            token = chunk.content
            respuesta_completa += token
            yield token

        self.memoria.agregar_turno(pregunta, respuesta_completa)
        self._actualizar_preferencias_desde_plan(plan, pregunta)

    # ── Gestión de memoria ────────────────────────────────────

    def limpiar_memoria(self) -> None:
        self.memoria.limpiar_sesion()
        print("🔄 Memoria EP2 limpiada.")

    def get_historial(self) -> list[dict]:
        msgs = self.memoria.obtener_historial_reciente()
        from langchain_core.messages import HumanMessage, AIMessage
        return [
            {"rol": "usuario" if isinstance(m, HumanMessage) else "bot",
             "texto": m.content}
            for m in msgs
        ]

    def estado_memoria(self) -> dict:
        return self.memoria.resumen_estado()

    # ── Actualización adaptativa de preferencias ─────────────

    def _actualizar_preferencias_desde_plan(self, plan, pregunta: str) -> None:
        """
        Toma de decisiones adaptativa (IL2.3):
        El planificador puede haber detectado intenciones clave que
        ameritan guardar información en la memoria larga.
        """
        from src.ep2_m7 import Intent
        if plan.intencion == Intent.CONFIRMACION:
            self.memoria.guardar_preferencia("etapa_venta", "confirmado")
        elif plan.intencion == Intent.ENVIO:
            self.memoria.guardar_preferencia("etapa_venta", "solicita_envio")


# ─────────────────────────────────────────────────────────────
# BLOQUE DE PRUEBA EN CONSOLA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agente = FutPlayersAgentEP2(ventana_memoria=4)

    escenarios = [
        # Escenario 1: flujo normal de venta
        ("Hola, ¿qué equipos tienen?",                 "saludo + catálogo"),
        ("¿Tienen polera del Real Madrid talla M?",    "stock talla específica"),
        ("¿Cuánto cuesta el envío a Temuco?",          "consulta envío"),
        ("Si compro 2 del mismo equipo, ¿hay descuento?", "cálculo descuento (CoT)"),
        ("Dale, las quiero. ¿Cómo pago?",              "confirmación + pago"),
        # Escenario 2: fuera de scope
        ("¿Cuál es la capital de Japón?",              "fuera de scope"),
    ]

    print("=" * 65)
    print("TEST FutPlayersAgentEP2 — EP2 ISY0101")
    print("=" * 65)

    for pregunta, descripcion in escenarios:
        print(f"\n[{descripcion.upper()}]")
        print(f"👤 {pregunta}")
        plan = agente.planificador.planificar(pregunta)
        print(f"   📋 Plan: {plan.intencion.value} | RAG={plan.requiere_rag} | CoT={plan.requiere_cot}")
        respuesta = agente.responder(pregunta)
        print(f"🤖 {respuesta[:200]}{'…' if len(respuesta) > 200 else ''}")
        print("-" * 55)

    print("\n📊 Estado final de memoria:")
    estado = agente.estado_memoria()
    print(f"   Turnos guardados: {estado['turnos_en_memoria']}")
    print(f"   Preferencias:     {estado['preferencias']}")