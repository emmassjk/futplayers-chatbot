"""
src/7_planificacion.py
======================
EP2 — IL2.3  Planificación y toma de decisiones del agente

Implementa un planificador de tareas basado en intención que:
  1. Clasifica la intención del usuario (INTENT_MAP)
  2. Asigna un plan de acción priorizado (TaskPlan)
  3. Ajusta el comportamiento del agente según condiciones del entorno
     (stock, descuentos, etapa de la venta)

Diagrama de flujo de decisión:
  Pregunta usuario
       │
       ▼
  Clasificar intención ──→ SALUDO / CATALOGO / STOCK / PRECIO
       │                    ENVIO / PAGO / DESCUENTO / CAMBIO
       │                    CONFIRMACION / FUERA_SCOPE
       ▼
  Construir TaskPlan
   ┌─────────────────────────────────────────────────────┐
   │  pasos:    lista ordenada de acciones               │
   │  prioridad: ALTA / MEDIA / BAJA                     │
   │  requiere_rag: bool                                 │
   │  requiere_cot: bool  (cadena de pensamiento)        │
   │  prompt_hint: texto extra inyectable en el prompt   │
   └─────────────────────────────────────────────────────┘
       │
       ▼
  Agente ejecuta plan → Respuesta final
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ─────────────────────────────────────────────────────────────
# 1. ENUMERACIÓN DE INTENCIONES
# ─────────────────────────────────────────────────────────────

class Intent(str, Enum):
    SALUDO         = "saludo"
    CATALOGO       = "catalogo"
    STOCK          = "stock"
    PRECIO         = "precio"
    ENVIO          = "envio"
    PAGO           = "pago"
    DESCUENTO      = "descuento"
    CAMBIO         = "cambio_devolucion"
    CONFIRMACION   = "confirmacion_compra"
    FUERA_SCOPE    = "fuera_scope"


# ─────────────────────────────────────────────────────────────
# 2. REGLAS DE CLASIFICACIÓN (keyword-based)
# ─────────────────────────────────────────────────────────────

# Cada tupla: (Intent, lista_de_keywords)
_REGLAS: list[tuple[Intent, list[str]]] = [
    (Intent.SALUDO,       ["hola", "buenas", "buenos días", "buenas tardes",
                           "buenas noches", "hey", "qué tal"]),
    (Intent.CONFIRMACION, ["si procedamos", "dale quiero", "lo tomo", "lo quiero",
                           "quiero comprar", "me la llevo", "okai", "sí, ",
                           "confirmado", "proceder", "sí quiero", "si quiero"]),
    (Intent.DESCUENTO,    ["descuento", "cuánto pagaría", "precio final", "total",
                           "cuántas", "comparar", "conviene", "calcul", "oferta",
                           "precio si compro", "si compro"]),
    (Intent.STOCK,        ["stock", "talla", "disponible", "hay", "tienen",
                           "queda", "quedan", "agotado"]),
    (Intent.PRECIO,       ["precio", "cuánto cuesta", "cuánto vale", "costo",
                           "valor", "cuánto sale"]),
    (Intent.ENVIO,        ["envío", "envio", "despacho", "starken", "llega",
                           "días", "región", "domicilio", "sucursal"]),
    (Intent.PAGO,         ["pago", "pagar", "transferencia", "tarjeta", "débito",
                           "crédito", "mercado pago", "efectivo", "cómo pago"]),
    (Intent.CAMBIO,       ["cambio", "devoluci", "talla equivocada", "no me queda",
                           "devolv", "retract"]),
    (Intent.CATALOGO,     ["catálogo", "catalogo", "qué equipos", "que equipos",
                           "todos", "opciones", "modelos", "disponibles",
                           "qué tienen", "que tienen", "muéstrame"]),
]

_KEYWORDS_FUERA: list[str] = [
    "política", "clima", "receta", "película", "chiste",
    "deporte que no sea fútbol", "basquetbol", "tenis", "bitcoin",
]


def clasificar_intencion(pregunta: str) -> Intent:
    """
    Clasifica la intención dominante de la pregunta.
    Orden de precedencia: CONFIRMACION > DESCUENTO > STOCK > resto.
    """
    texto = pregunta.lower()

    # Fuera de scope explícito
    if any(k in texto for k in _KEYWORDS_FUERA):
        return Intent.FUERA_SCOPE

    for intencion, keywords in _REGLAS:
        if any(k in texto for k in keywords):
            return intencion

    # Default: asumir consulta de catálogo si no hay señal clara
    return Intent.CATALOGO


# ─────────────────────────────────────────────────────────────
# 3. PLAN DE TAREAS
# ─────────────────────────────────────────────────────────────

class Prioridad(str, Enum):
    ALTA  = "ALTA"
    MEDIA = "MEDIA"
    BAJA  = "BAJA"


@dataclass
class TaskPlan:
    """
    Estructura que describe qué debe hacer el agente para responder.

    Atributos:
        intencion    : intención detectada
        pasos        : lista ordenada de acciones a ejecutar
        prioridad    : nivel de urgencia del plan
        requiere_rag : si debe recuperar contexto del vector store
        requiere_cot : si debe usar cadena de pensamiento (cálculos)
        prompt_hint  : texto adicional para inyectar en el system prompt
    """
    intencion:     Intent
    pasos:         list[str]     = field(default_factory=list)
    prioridad:     Prioridad     = Prioridad.MEDIA
    requiere_rag:  bool          = True
    requiere_cot:  bool          = False
    prompt_hint:   str           = ""


# ─────────────────────────────────────────────────────────────
# 4. PLANIFICADOR — construye TaskPlan según intención
# ─────────────────────────────────────────────────────────────

class Planificador:
    """
    Motor de planificación del agente EP2.

    Dado un texto de entrada, devuelve un TaskPlan con los pasos
    y configuraciones necesarias para responder adecuadamente.

    Estrategia de ajuste adaptativo:
      • Si la intención es CONFIRMACION → prioridad ALTA, no RAG, pasos de cierre.
      • Si la intención es DESCUENTO    → CoT obligatorio, RAG para precios.
      • Si la intención es FUERA_SCOPE  → sin RAG, respuesta directa de rechazo.
    """

    _PLANES: dict[Intent, Callable[[], TaskPlan]] = {}  # se registran abajo

    def planificar(self, pregunta: str) -> TaskPlan:
        intencion = clasificar_intencion(pregunta)
        builder   = self._PLANES.get(intencion, self._plan_default)
        plan      = builder()
        plan.intencion = intencion
        return plan

    # ── Builders de planes ──────────────────────────────────

    @staticmethod
    def _plan_saludo() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.SALUDO,
            pasos=["1. Saludar con nombre del bot",
                   "2. Listar brevemente servicios disponibles",
                   "3. Invitar al cliente a preguntar"],
            prioridad=Prioridad.BAJA,
            requiere_rag=False,
            prompt_hint="Responde con el saludo estándar de Futbot.",
        )

    @staticmethod
    def _plan_catalogo() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.CATALOGO,
            pasos=["1. Consultar RAG por equipo o categoría",
                   "2. Mostrar productos relevantes con precio y tallas",
                   "3. Invitar a especificar talla o confirmar compra"],
            prioridad=Prioridad.MEDIA,
            requiere_rag=True,
            prompt_hint="Usa el formato de producto estándar (👕 Nombre, 📏 Tallas, 💰 Precio).",
        )

    @staticmethod
    def _plan_stock() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.STOCK,
            pasos=["1. Recuperar producto del RAG",
                   "2. Verificar stock por talla en metadatos",
                   "3. Informar disponibilidad o sugerir alternativa"],
            prioridad=Prioridad.ALTA,
            requiere_rag=True,
            prompt_hint="Revisa el campo 'Stock por talla' del contexto. Si stock=0, propón alternativa.",
        )

    @staticmethod
    def _plan_precio() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.PRECIO,
            pasos=["1. Recuperar precio del contexto RAG",
                   "2. Presentar precio con formato $XX.000 CLP",
                   "3. Mencionar envío gratis si aplica (>$39.990)"],
            prioridad=Prioridad.MEDIA,
            requiere_rag=True,
            prompt_hint="Nunca inventes precios. Usa los del contexto.",
        )

    @staticmethod
    def _plan_descuento() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.DESCUENTO,
            pasos=["1. Identificar productos y cantidad",
                   "2. Calcular descuento aplicable (10% o 15%)",
                   "3. Calcular total con descuento",
                   "4. Verificar si aplica envío gratis (>$39.990)",
                   "5. Mostrar resumen con ahorro destacado"],
            prioridad=Prioridad.ALTA,
            requiere_rag=True,
            requiere_cot=True,
            prompt_hint="Razona paso a paso antes de entregar el precio final.",
        )

    @staticmethod
    def _plan_envio() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.ENVIO,
            pasos=["1. Consultar políticas de envío del RAG",
                   "2. Informar plazos según región",
                   "3. Indicar si aplica envío gratis"],
            prioridad=Prioridad.MEDIA,
            requiere_rag=True,
            prompt_hint="RM: 1-2 días. Regiones: 2-5 días. Gratis sobre $39.990.",
        )

    @staticmethod
    def _plan_pago() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.PAGO,
            pasos=["1. Informar que solo se acepta transferencia bancaria",
                   "2. Entregar datos bancarios completos",
                   "3. Indicar dónde enviar comprobante"],
            prioridad=Prioridad.MEDIA,
            requiere_rag=False,
            prompt_hint="SOLO transferencia. Incluir datos bancarios completos.",
        )

    @staticmethod
    def _plan_cambio() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.CAMBIO,
            pasos=["1. Consultar política de cambios del RAG",
                   "2. Informar plazo (10 días corridos)",
                   "3. Indicar condiciones y contacto"],
            prioridad=Prioridad.BAJA,
            requiere_rag=True,
            prompt_hint="Cambio dentro de 10 días, sin uso, con embalaje original.",
        )

    @staticmethod
    def _plan_confirmacion() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.CONFIRMACION,
            pasos=["1. Mostrar resumen completo del pedido",
                   "2. Calcular total (con descuentos si aplica)",
                   "3. Entregar datos de transferencia",
                   "4. Preguntar modalidad de entrega (envío/retiro)",
                   "5. Confirmar pedido y agradecer"],
            prioridad=Prioridad.ALTA,
            requiere_rag=False,
            requiere_cot=True,
            prompt_hint="El cliente está listo para comprar. Cierra la venta ahora.",
        )

    @staticmethod
    def _plan_fuera_scope() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.FUERA_SCOPE,
            pasos=["1. Indicar cortésmente que el tema está fuera del alcance",
                   "2. Redirigir a temas de FutPlayers.cl"],
            prioridad=Prioridad.BAJA,
            requiere_rag=False,
            prompt_hint="Solo habla de FutPlayers.cl y fútbol.",
        )

    @staticmethod
    def _plan_default() -> TaskPlan:
        return TaskPlan(
            intencion=Intent.CATALOGO,
            pasos=["1. Recuperar contexto relevante del RAG",
                   "2. Responder la consulta"],
            prioridad=Prioridad.MEDIA,
            requiere_rag=True,
        )


# Registro de builders
Planificador._PLANES = {
    Intent.SALUDO:       Planificador._plan_saludo,
    Intent.CATALOGO:     Planificador._plan_catalogo,
    Intent.STOCK:        Planificador._plan_stock,
    Intent.PRECIO:       Planificador._plan_precio,
    Intent.DESCUENTO:    Planificador._plan_descuento,
    Intent.ENVIO:        Planificador._plan_envio,
    Intent.PAGO:         Planificador._plan_pago,
    Intent.CAMBIO:       Planificador._plan_cambio,
    Intent.CONFIRMACION: Planificador._plan_confirmacion,
    Intent.FUERA_SCOPE:  Planificador._plan_fuera_scope,
}


# ─────────────────────────────────────────────────────────────
# BLOQUE DE PRUEBA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    planificador = Planificador()

    casos = [
        "Hola, ¿qué equipos tienen?",
        "¿Tienen polera del Real Madrid talla M?",
        "Si compro 3 poleras de distintos equipos, ¿me hacen descuento?",
        "Dale quiero las dos, ¿cómo pago?",
        "¿Puedo pagar con tarjeta de crédito?",
        "¿Puedo devolver si no me queda?",
        "¿Cuál es la capital de Francia?",
    ]

    print("=" * 60)
    print("TEST Planificador FutPlayers EP2")
    print("=" * 60)

    for pregunta in casos:
        plan = planificador.planificar(pregunta)
        print(f"\n❓ '{pregunta}'")
        print(f"   Intención   : {plan.intencion.value}")
        print(f"   Prioridad   : {plan.prioridad.value}")
        print(f"   RAG         : {plan.requiere_rag}  |  CoT: {plan.requiere_cot}")
        print(f"   Pasos       :")
        for paso in plan.pasos:
            print(f"     {paso}")