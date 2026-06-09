"""
=============================================================================
  SISTEMA DE AGENTES PARA RESTAURANTE --- Proyecto LangGraph + OpenAI
=============================================================================
  Arquitectura: StateGraph cíclico con nodos agent, ver_menu y crear_reserva.
  Flujo:
    __start__ -> agent -> [router] -> ver_menu      -> agent -> END
                                   -> crear_reserva -> agent -> END  ← INTERRUPT antes
                                   -> END

  Human-in-the-Loop (HITL):
    - MemorySaver:         persiste el estado entre turnos (memoria de conversación).
    - interrupt_before:    pausa el grafo ANTES de ejecutar crear_reserva para
                           que el usuario apruebe o rechace la reserva.
=============================================================================
"""

# ─────────────────────────────────────────────
# 1. IMPORTACIONES
# ─────────────────────────────────────────────
import sys
import io
if getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver          # ← NUEVO: checkpointer
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import AzureChatOpenAI

import datetime
import random

# ─────────────────────────────────────────────
# 2. CARGAR VARIABLES DE ENTORNO DESDE .env
# ─────────────────────────────────────────────
load_dotenv()

AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY",  "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
OPENAI_TEMP      = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

if not AZURE_API_KEY or AZURE_API_KEY == "PON_AQUI_TU_API_KEY_DE_FOUNDRY":
    raise EnvironmentError(
        "\n❌ ERROR: AZURE_OPENAI_API_KEY no está configurada.\n"
        "   Abre el archivo .env y pega la API Key de Microsoft Foundry.\n"
    )
if not AZURE_ENDPOINT or "TU_ENDPOINT" in AZURE_ENDPOINT:
    raise EnvironmentError(
        "\n❌ ERROR: AZURE_OPENAI_ENDPOINT no está configurado.\n"
        "   Abre el archivo .env y pega el Project endpoint de Foundry.\n"
    )

print(f"✅ Azure OpenAI configurado:")
print(f"   Deployment : {AZURE_DEPLOYMENT}")
print(f"   Endpoint   : {AZURE_ENDPOINT}")
print(f"   API Version: {AZURE_API_VER}")
print(f"   Temperatura: {OPENAI_TEMP}")

# ─────────────────────────────────────────────
# 3. INSTANCIA DEL LLM
# ─────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    azure_deployment=AZURE_DEPLOYMENT,
    api_version=AZURE_API_VER,
    temperature=OPENAI_TEMP,
)

# ─────────────────────────────────────────────
# 4. DEFINICIÓN DEL ESTADO COMPARTIDO
# ─────────────────────────────────────────────
class EstadoRestaurante(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ─────────────────────────────────────────────
# 5. BASE DE DATOS SIMULADA DEL RESTAURANTE
# ─────────────────────────────────────────────
MENU_DEL_DIA = {
    "entrantes": [
        {"nombre": "Gazpacho Andaluz",        "precio": 6.50,  "alergenos": "sin gluten"},
        {"nombre": "Croquetas de Jamón (6u)", "precio": 8.00,  "alergenos": "gluten, lácteos"},
        {"nombre": "Ensalada Mixta",          "precio": 5.50,  "alergenos": "sin gluten"},
    ],
    "principales": [
        {"nombre": "Merluza a la Romana",     "precio": 15.00, "alergenos": "gluten, pescado"},
        {"nombre": "Solomillo con Patatas",   "precio": 18.50, "alergenos": "sin gluten"},
        {"nombre": "Pasta Carbonara",         "precio": 12.00, "alergenos": "gluten, lácteos, huevo"},
    ],
    "postres": [
        {"nombre": "Tarta de Queso",          "precio": 5.00,  "alergenos": "lácteos, gluten"},
        {"nombre": "Crème Brûlée",            "precio": 5.50,  "alergenos": "lácteos, huevo"},
        {"nombre": "Sorbete de Limón",        "precio": 4.00,  "alergenos": "sin alérgenos comunes"},
    ],
}

RESERVAS_REGISTRADAS: list[dict] = []

SYSTEM_PROMPT = """Eres el asistente virtual del Restaurante "La Buena Mesa", un restaurante
español de alta cocina. Tu nombre es MesaBot.

Tus capacidades son:
  1. Mostrar el menú del día con precios y alérgenos.
  2. Crear reservas para los clientes (requieren confirmación del cliente antes de procesarse).

Cuando recibas los datos de una herramienta (menú o reserva confirmada),
preséntaselos al cliente de forma clara, amigable y en español.
Si recibes [RESERVA_CANCELADA], informa amablemente al cliente de que la reserva
no se ha registrado y ofrécele ayuda para cuando quiera intentarlo de nuevo.
Si el cliente solo saluda o hace una pregunta general, responde brevemente
y ofrécele las dos opciones disponibles (ver menú / hacer reserva).
Cuando el cliente quiera hacer una reserva, resume los datos que has entendido
(personas, fecha, hora) y dile que vas a solicitar su confirmación antes de registrarla.
Mantén un tono profesional pero cercano. Sé conciso."""


# ─────────────────────────────────────────────
# 6. NODO: "agent"
# ─────────────────────────────────────────────
def nodo_agent(estado: EstadoRestaurante) -> EstadoRestaurante:
    print("\n" + "="*60)
    print("  🧠  NODO: agent  (consultando GPT-4o-mini...)")
    print("="*60)

    mensajes = estado["messages"]
    print(f"  📊 Mensajes en historial: {len(mensajes)}")
    ultimo = mensajes[-1].content
    print(f"  📥 Último mensaje: '{ultimo[:70]}...'" if len(ultimo) > 70 else f"  📥 Último mensaje: '{ultimo}'")

    mensajes_llm = [SystemMessage(content=SYSTEM_PROMPT)] + mensajes
    respuesta_llm = llm.invoke(mensajes_llm)

    resp = respuesta_llm.content
    print(f"  📤 Respuesta del LLM: '{resp[:70]}...'" if len(resp) > 70 else f"  📤 Respuesta: '{resp}'")

    return {"messages": [respuesta_llm]}


# ─────────────────────────────────────────────
# 7. NODO: "ver_menu"
# ─────────────────────────────────────────────
def nodo_ver_menu(estado: EstadoRestaurante) -> EstadoRestaurante:
    print("\n" + "-"*60)
    print("  🍽️   NODO: ver_menu  (consultando base de datos...)")
    print("-"*60)

    lineas = [f"DATOS DEL MENÚ DEL DÍA ({datetime.date.today().strftime('%d/%m/%Y')}):\n"]
    for categoria, platos in MENU_DEL_DIA.items():
        lineas.append(f"\n{categoria.upper()}:")
        for plato in platos:
            lineas.append(
                f"  - {plato['nombre']}: {plato['precio']:.2f}€  "
                f"[Alérgenos: {plato['alergenos']}]"
            )

    resultado = "\n".join(lineas)
    print(f"  ✅ Menú generado ({len(MENU_DEL_DIA)} categorías, "
          f"{sum(len(v) for v in MENU_DEL_DIA.values())} platos)")

    return {"messages": [AIMessage(content=f"[MENU]\n{resultado}")]}


# ─────────────────────────────────────────────
# 8. NODO: "crear_reserva"
# ─────────────────────────────────────────────
# ⚠️  Este nodo tiene interrupt_before: el grafo se PAUSA antes de ejecutarlo
#     y espera la aprobación humana desde chat_interactivo.py.
def nodo_crear_reserva(estado: EstadoRestaurante) -> EstadoRestaurante:
    print("\n" + "-"*60)
    print("  📅  NODO: crear_reserva  (aprobado — procesando reserva...)")
    print("-"*60)

    mensajes = estado["messages"]

    # ── Usar el ÚLTIMO HumanMessage para extraer datos ──
    # Con memoria multi-turno puede haber varios HumanMessages; queremos el más reciente.
    mensaje_usuario = next(
        (m.content for m in reversed(mensajes) if isinstance(m, HumanMessage)),
        "Reserva para 2 personas"
    )

    texto = mensaje_usuario.lower()

    personas = 2
    for num, palabras in [
        (1, ["una persona", "1 persona", " uno "]),
        (2, ["dos personas", "2 personas", " dos "]),
        (3, ["tres personas", "3 personas", " tres "]),
        (4, ["cuatro personas", "4 personas", " cuatro "]),
        (5, ["cinco personas", "5 personas", " cinco "]),
        (6, ["seis personas", "6 personas", " seis "]),
    ]:
        if any(p in texto for p in palabras):
            personas = num
            break

    if "mañana" in texto:
        fecha = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%d/%m/%Y")
    elif "hoy" in texto:
        fecha = datetime.date.today().strftime("%d/%m/%Y")
    else:
        fecha = (datetime.date.today() + datetime.timedelta(days=2)).strftime("%d/%m/%Y")

    if any(p in texto for p in ["mediodía", "mediodia", "comida", "14:00", "las 14"]):
        hora = "14:00"
    elif any(p in texto for p in ["noche", "cena", "21:00", "las 21", "20:00", "las 20"]):
        hora = "21:00"
    else:
        hora = "13:30"

    codigo_reserva = f"RES-{random.randint(1000, 9999)}"
    nueva_reserva = {
        "codigo":   codigo_reserva,
        "personas": personas,
        "fecha":    fecha,
        "hora":     hora,
        "estado":   "confirmada",
    }
    RESERVAS_REGISTRADAS.append(nueva_reserva)

    print(f"  ✅ Reserva registrada: {nueva_reserva}")
    print(f"  📋 Total reservas en sistema: {len(RESERVAS_REGISTRADAS)}")

    confirmacion = (
        f"RESERVA CONFIRMADA EN EL SISTEMA:\n"
        f"  - Código:   {codigo_reserva}\n"
        f"  - Personas: {personas}\n"
        f"  - Fecha:    {fecha}\n"
        f"  - Hora:     {hora}\n"
        f"  - Estado:   confirmada\n"
        f"El cliente recibirá un recordatorio el día anterior."
    )

    return {"messages": [AIMessage(content=f"[RESERVA]\n{confirmacion}")]}


# ─────────────────────────────────────────────
# 9. ROUTER CONDICIONAL
# ─────────────────────────────────────────────
def router_condicional(estado: EstadoRestaurante) -> Literal["ver_menu", "crear_reserva", "__end__"]:
    print("\n" + "·"*60)
    print("  🔀  ROUTER: evaluando destino del flujo...")
    print("·"*60)

    mensajes = estado["messages"]

    # ── CLAVE ANTI-BUCLE con soporte multi-turno ──
    # Solo miramos los mensajes DESDE el último HumanMessage (turno actual).
    # Así no confundimos herramientas de turnos anteriores con el turno actual.
    last_human_idx = max(
        (i for i, m in enumerate(mensajes) if isinstance(m, HumanMessage)),
        default=0
    )
    mensajes_turno_actual = mensajes[last_human_idx:]

    herramienta_ejecutada = any(
        isinstance(m, AIMessage) and (
            m.content.startswith("[MENU]")
            or m.content.startswith("[RESERVA]")
            or m.content.startswith("[RESERVA_CANCELADA]")
        )
        for m in mensajes_turno_actual
    )

    if herramienta_ejecutada:
        print("  ℹ️  Herramienta ya ejecutada en este turno. Ciclo completado.")
        print("  ✅ Decisión: → __end__")
        return "__end__"

    # ── Extraer la intención del turno actual (último HumanMessage) ──
    mensaje_humano = next(
        (m.content for m in reversed(mensajes) if isinstance(m, HumanMessage)), ""
    )
    texto = mensaje_humano.lower()

    print(f"  🔍 Intención: '{mensaje_humano[:65]}...'" if len(mensaje_humano) > 65
          else f"  🔍 Intención: '{mensaje_humano}'")

    palabras_menu = [
        "menú", "menu", "carta", "platos", "comer", "comida",
        "que tienen", "qué tienen", "tienen", "ofrecen", "opciones",
        "qué hay", "que hay", "ver el menú", "mostrar"
    ]
    palabras_reserva = [
        "reserva", "reservar", "mesa", "booking", "apartar",
        "lugar", "sitio", "noche", "persona", "personas",
        "para dos", "para tres", "para cuatro", "quiero una mesa"
    ]

    if any(p in texto for p in palabras_menu):
        print("  ✅ Decisión: → ver_menu")
        return "ver_menu"
    elif any(p in texto for p in palabras_reserva):
        print("  ✅ Decisión: → crear_reserva")
        return "crear_reserva"
    else:
        print("  ✅ Decisión: → __end__  (sin herramienta necesaria)")
        return "__end__"


# ─────────────────────────────────────────────
# 10. CONSTRUCCIÓN DEL GRAFO
# ─────────────────────────────────────────────
def construir_grafo():
    """
    Construye, configura y compila el StateGraph del restaurante.

    Novedades HITL:
      - MemorySaver como checkpointer: persiste el estado entre llamadas a invoke().
        Cada conversación se identifica por su thread_id en el config.
      - interrupt_before=["crear_reserva"]: el grafo se pausa justo antes de
        ejecutar nodo_crear_reserva y devuelve el control al llamador (chat_interactivo)
        para que el usuario apruebe o rechace la reserva.

    Estructura:
      START → agent → [router] → ver_menu      → agent → END
                               → crear_reserva → agent → END   (⏸ PAUSA aquí)
                               → END
    """
    print("\n🔧 Construyendo el grafo LangGraph con Human-in-the-Loop...")

    grafo = StateGraph(EstadoRestaurante)

    grafo.add_node("agent",         nodo_agent)
    grafo.add_node("ver_menu",      nodo_ver_menu)
    grafo.add_node("crear_reserva", nodo_crear_reserva)

    grafo.add_edge(START,           "agent")
    grafo.add_edge("ver_menu",      "agent")
    grafo.add_edge("crear_reserva", "agent")

    grafo.add_conditional_edges(
        "agent",
        router_condicional,
        {
            "ver_menu":      "ver_menu",
            "crear_reserva": "crear_reserva",
            "__end__":       END,
        }
    )

    # ── Checkpointer: guarda el estado completo en memoria entre invocaciones ──
    memory = MemorySaver()

    # ── interrupt_before: pausa el grafo ANTES de crear_reserva ──
    app = grafo.compile(
        checkpointer=memory,
        interrupt_before=["crear_reserva"],
    )

    print("✅ Grafo compilado con HITL (MemorySaver + interrupt_before=[crear_reserva]).\n")
    return app
