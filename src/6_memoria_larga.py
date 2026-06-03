"""
src/6_memoria_larga.py
======================
EP2 — IL2.2  Memoria de corto y largo plazo para FutPlayersAgent

Arquitectura de memoria implementada:
┌──────────────────────────────────────────────────────────────┐
│  CORTO PLAZO  →  chat_history  (lista HumanMessage/AIMessage)│
│               →  ventana deslizante de N últimos turnos      │
│                                                              │
│  LARGO PLAZO  →  MemorySaver (LangGraph checkpoint en RAM)   │
│               →  thread_id identifica cada sesión            │
│               →  SessionStore persiste preferencias clave    │
└──────────────────────────────────────────────────────────────┘

Cómo se usa desde 7_agente_ep2.py:
    from src.6_memoria_larga import MemoryManager
    mm = MemoryManager(ventana_corto_plazo=6)
    mm.agregar_turno("¿Tienen polera del Madrid?", "Sí, tenemos…")
    contexto_corto = mm.obtener_historial_reciente()
    mm.guardar_preferencia("equipo_favorito", "Real Madrid")
    pref = mm.obtener_preferencia("equipo_favorito")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ─────────────────────────────────────────────────────────────
# 1. MEMORIA DE CORTO PLAZO — ventana deslizante
# ─────────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Almacena los últimos `ventana` pares (humano, bot) de la sesión activa.
    Garantiza continuidad en flujos prolongados sin saturar el contexto del LLM.
    """

    def __init__(self, ventana: int = 6):
        self.ventana: int = ventana          # turnos a conservar (1 turno = 1 par H+A)
        self._mensajes: list[BaseMessage] = []

    def agregar(self, pregunta: str, respuesta: str) -> None:
        self._mensajes.append(HumanMessage(content=pregunta))
        self._mensajes.append(AIMessage(content=respuesta))
        # Recortar: conservar solo los últimos ventana*2 mensajes
        max_msgs = self.ventana * 2
        if len(self._mensajes) > max_msgs:
            self._mensajes = self._mensajes[-max_msgs:]

    def obtener(self) -> list[BaseMessage]:
        return list(self._mensajes)

    def limpiar(self) -> None:
        self._mensajes = []

    def __len__(self) -> int:
        return len(self._mensajes) // 2   # número de turnos


# ─────────────────────────────────────────────────────────────
# 2. MEMORIA DE LARGO PLAZO — preferencias de sesión
# ─────────────────────────────────────────────────────────────

class LongTermMemory:
    """
    Persiste preferencias semánticas del usuario entre turnos de la misma sesión.
    Implementación: diccionario serializado a JSON en memoria (extensible a disco).

    Claves semánticas típicas:
        "equipo_favorito"   → "Real Madrid"
        "talla"             → "M"
        "comuna_envio"      → "Providencia"
        "tipo_envio"        → "domicilio"
    """

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}

    def guardar(self, clave: str, valor: Any) -> None:
        self._store[clave] = valor
        self._timestamps[clave] = time.time()

    def obtener(self, clave: str, default: Any = None) -> Any:
        return self._store.get(clave, default)

    def todas(self) -> dict[str, Any]:
        return dict(self._store)

    def limpiar(self) -> None:
        self._store.clear()
        self._timestamps.clear()

    def exportar_json(self) -> str:
        return json.dumps(self._store, ensure_ascii=False, indent=2)

    def importar_json(self, datos: str) -> None:
        self._store = json.loads(datos)

    def __repr__(self) -> str:
        return f"LongTermMemory({self._store})"


# ─────────────────────────────────────────────────────────────
# 3. MEMORY MANAGER — orquesta ambas memorias
# ─────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Punto de entrada único para el agente EP2.

    Orquesta:
      • ShortTermMemory  → historial conversacional reciente
      • LongTermMemory   → preferencias semánticas persistentes

    Detección automática de preferencias:
      Si la respuesta del bot contiene datos del usuario (talla, equipo, comuna),
      extrae y persiste automáticamente esos valores.
    """

    # Patrones de extracción semántica (keywords → clave de preferencia)
    _PATRONES_TALLA = {"talla s": "S", "talla m": "M", "talla l": "L", "talla xl": "XL"}
    _ALIAS_EQUIPOS  = {
        "colo-colo": "Colo-Colo", "colo colo": "Colo-Colo",
        "la u": "Universidad de Chile", "u de chile": "Universidad de Chile",
        "u. de chile": "Universidad de Chile",
        "católica": "Universidad Católica", "real madrid": "Real Madrid",
        "barcelona": "Barcelona", "psg": "PSG",
    }

    def __init__(self, ventana_corto_plazo: int = 6):
        self.corto = ShortTermMemory(ventana=ventana_corto_plazo)
        self.largo = LongTermMemory()

    # ── API principal ──────────────────────────────────────────

    def agregar_turno(self, pregunta: str, respuesta: str) -> None:
        """Registra un turno completo y extrae preferencias automáticamente."""
        self.corto.agregar(pregunta, respuesta)
        self._extraer_preferencias(pregunta)

    def obtener_historial_reciente(self) -> list[BaseMessage]:
        """Devuelve el historial de corto plazo para inyectar en el prompt."""
        return self.corto.obtener()

    def guardar_preferencia(self, clave: str, valor: Any) -> None:
        self.largo.guardar(clave, valor)

    def obtener_preferencia(self, clave: str, default: Any = None) -> Any:
        return self.largo.obtener(clave, default)

    def enriquecer_contexto(self) -> str:
        """
        Genera texto de contexto semántico para inyectar en el prompt.
        Permite que el agente recuerde preferencias del usuario sin repetir
        la pregunta manualmente.
        """
        prefs = self.largo.todas()
        if not prefs:
            return ""
        lineas = ["[Preferencias conocidas del cliente en esta sesión]"]
        mapping = {
            "equipo_favorito": "Equipo favorito",
            "talla":           "Talla preferida",
            "comuna_envio":    "Comuna de envío",
            "tipo_envio":      "Tipo de envío",
        }
        for clave, label in mapping.items():
            if clave in prefs:
                lineas.append(f"  • {label}: {prefs[clave]}")
        return "\n".join(lineas)

    def limpiar_sesion(self) -> None:
        """Reinicia ambas memorias (nueva conversación)."""
        self.corto.limpiar()
        self.largo.limpiar()

    def resumen_estado(self) -> dict:
        return {
            "turnos_en_memoria":    len(self.corto),
            "ventana_configurada":  self.corto.ventana,
            "preferencias":         self.largo.todas(),
        }

    # ── Extracción automática ──────────────────────────────────

    def _extraer_preferencias(self, texto: str) -> None:
        t = texto.lower()
        # Talla
        for patron, valor in self._PATRONES_TALLA.items():
            if patron in t:
                self.largo.guardar("talla", valor)
                break
        # Equipo
        for alias, equipo in self._ALIAS_EQUIPOS.items():
            if alias in t:
                self.largo.guardar("equipo_favorito", equipo)
                break


# ─────────────────────────────────────────────────────────────
# BLOQUE DE PRUEBA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("TEST MemoryManager — FutPlayers EP2")
    print("=" * 55)

    mm = MemoryManager(ventana_corto_plazo=3)

    conversacion = [
        ("¿Tienen polera del Colo-Colo talla M?",  "Sí, tenemos varias opciones."),
        ("¿Cuánto cuesta el envío a Temuco?",       "A Temuco tarda 2-5 días hábiles."),
        ("¿Hay descuento si compro 2?",             "Sí, 10% en la segunda unidad."),
        ("OK dale, la quiero",                      "¡Perfecto! ¿Envío a domicilio o retiro?"),
        ("Envío a domicilio",                       "Te paso el formulario…"),
    ]

    for pregunta, respuesta in conversacion:
        mm.agregar_turno(pregunta, respuesta)

    estado = mm.resumen_estado()
    print(f"\nTurnos en memoria (ventana=3): {estado['turnos_en_memoria']}")
    print(f"Preferencias detectadas:       {estado['preferencias']}")
    print(f"\nContexto semántico inyectable:\n{mm.enriquecer_contexto()}")
    print(f"\nHistorial reciente ({len(mm.obtener_historial_reciente())} msgs):")
    for m in mm.obtener_historial_reciente():
        rol = "👤 H" if isinstance(m, HumanMessage) else "🤖 A"
        print(f"  {rol}: {m.content[:60]}…")