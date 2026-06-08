"""
=============================================================================
  SISTEMA DE AGENTES PARA RESTAURANTE --- Proyecto LangGraph + OpenAI
=============================================================================
  Arquitectura: StateGraph ciclico con nodos agent, ver_menu y crear_reserva.
  Flujo:
    __start__ -> agent -> [router] -> ver_menu      -> agent -> END
                                   -> crear_reserva -> agent -> END
                                   -> END

  El nodo "agent" usa GPT-4o-mini como cerebro para generar respuestas
  naturales. Las herramientas son nodos puros de Python.
=============================================================================
"""

# ─────────────────────────────────────────────
# 1. IMPORTACIONES
# ─────────────────────────────────────────────
import sys
import io
# Forzar UTF-8 en stdout para que los emojis funcionen en Windows (cp1252)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
from typing import TypedDict, Annotated, Literal

# python-dotenv: carga el archivo .env antes de leer os.environ
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import AzureChatOpenAI

import datetime
import random

# ─────────────────────────────────────────────
# 2. CARGAR VARIABLES DE ENTORNO DESDE .env
# ─────────────────────────────────────────────
# load_dotenv() busca el archivo .env en el directorio actual y
# carga las variables como variables de entorno del proceso.
load_dotenv()

# Variables específicas de Azure OpenAI (Microsoft Foundry)
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY",  "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
OPENAI_TEMP      = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# Validación de credenciales antes de arrancar el grafo
if not AZURE_API_KEY or AZURE_API_KEY == "PON_AQUI_TU_API_KEY_DE_FOUNDRY":
    raise EnvironmentError(
        "\n❌ ERROR: AZURE_OPENAI_API_KEY no está configurada.\n"
        "   Abre el archivo .env y pega la API Key que ves\n"
        "   en Microsoft Foundry → tu deployment → 'API Key'.\n"
    )
if not AZURE_ENDPOINT or "TU_ENDPOINT" in AZURE_ENDPOINT:
    raise EnvironmentError(
        "\n❌ ERROR: AZURE_OPENAI_ENDPOINT no está configurado.\n"
        "   Abre el archivo .env y pega el 'Project endpoint'\n"
        "   que aparece en Foundry (ej: https://oscar23-resource.services.ai.azure.com/)\n"
    )

print(f"✅ Azure OpenAI configurado:")
print(f"   Deployment : {AZURE_DEPLOYMENT}")
print(f"   Endpoint   : {AZURE_ENDPOINT}")
print(f"   API Version: {AZURE_API_VER}")
print(f"   Temperatura: {OPENAI_TEMP}")

# ─────────────────────────────────────────────
# 3. INSTANCIA DEL LLM (Azure GPT-4o-mini)
# ─────────────────────────────────────────────
# AzureChatOpenAI conecta con tu deployment en Microsoft Foundry.
# Requiere endpoint, api_key, deployment name y api_version.
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
# El Estado es la "memoria" que viaja entre todos los nodos del grafo.
# - 'messages' usa Annotated con add_messages para ACUMULAR mensajes
#   en lugar de sobrescribirlos en cada paso.
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

# Prompt de sistema: define la personalidad y las capacidades del agente
SYSTEM_PROMPT = """Eres el asistente virtual del Restaurante "La Buena Mesa", un restaurante
español de alta cocina. Tu nombre es MesaBot.

Tus capacidades son:
  1. Mostrar el menú del día con precios y alérgenos.
  2. Crear reservas para los clientes.

Cuando recibas los datos de una herramienta (menú o reserva confirmada),
preséntaselos al cliente de forma clara, amigable y en español.
Si el cliente solo saluda o hace una pregunta general, responde brevemente
y ofrécele las dos opciones disponibles (ver menú / hacer reserva).
Mantén un tono profesional pero cercano. Sé conciso."""


# ─────────────────────────────────────────────
# 6. NODO: "agent" — El Cerebro del Sistema
# ─────────────────────────────────────────────
# Envía el historial completo de mensajes al LLM y devuelve
# la respuesta generada como un nuevo AIMessage en el estado.
def nodo_agent(estado: EstadoRestaurante) -> EstadoRestaurante:
    print("\n" + "="*60)
    print("  🧠  NODO: agent  (consultando GPT-4o-mini...)")
    print("="*60)

    mensajes = estado["messages"]
    print(f"  📊 Mensajes en historial: {len(mensajes)}")
    print(f"  📥 Último mensaje: '{mensajes[-1].content[:70]}...' "
          if len(mensajes[-1].content) > 70 else f"  📥 Último mensaje: '{mensajes[-1].content}'")

    # Construir la lista de mensajes para el LLM:
    # SystemMessage define el comportamiento global del agente.
    # El resto son el historial de la conversación.
    mensajes_llm = [SystemMessage(content=SYSTEM_PROMPT)] + mensajes

    # Llamada real al LLM — aquí viaja la petición a la API de OpenAI
    respuesta_llm = llm.invoke(mensajes_llm)

    print(f"  📤 Respuesta del LLM: '{respuesta_llm.content[:70]}...' "
          if len(respuesta_llm.content) > 70 else f"  📤 Respuesta: '{respuesta_llm.content}'")

    return {"messages": [respuesta_llm]}


# ─────────────────────────────────────────────
# 7. NODO: "ver_menu" — Herramienta de Menú
# ─────────────────────────────────────────────
# Consulta la BD simulada y construye el texto del menú.
# SIEMPRE retorna al nodo "agent" (nunca a END directamente).
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
    print(f"  ↩️  Retornando datos al nodo 'agent'...")

    # Prefijo [MENU] para que el router detecte que la herramienta ya se ejecutó
    return {"messages": [AIMessage(content=f"[MENU]\n{resultado}")]}


# ─────────────────────────────────────────────
# 8. NODO: "crear_reserva" — Herramienta de Reservas
# ─────────────────────────────────────────────
# Extrae datos del mensaje del usuario y registra la reserva en memoria.
# SIEMPRE retorna al nodo "agent" (nunca a END directamente).
def nodo_crear_reserva(estado: EstadoRestaurante) -> EstadoRestaurante:
    print("\n" + "-"*60)
    print("  📅  NODO: crear_reserva  (procesando reserva...)")
    print("-"*60)

    mensajes = estado["messages"]
    mensaje_usuario = next(
        (m.content for m in mensajes if isinstance(m, HumanMessage)),
        "Reserva para 2 personas"
    )

    # Extracción de datos (NLP básico simulado)
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
    print(f"  ↩️  Retornando confirmación al nodo 'agent'...")

    confirmacion = (
        f"RESERVA CONFIRMADA EN EL SISTEMA:\n"
        f"  - Código:   {codigo_reserva}\n"
        f"  - Personas: {personas}\n"
        f"  - Fecha:    {fecha}\n"
        f"  - Hora:     {hora}\n"
        f"  - Estado:   confirmada\n"
        f"El cliente recibirá un recordatorio el día anterior."
    )

    # Prefijo [RESERVA] para que el router detecte que ya se ejecutó esta herramienta
    return {"messages": [AIMessage(content=f"[RESERVA]\n{confirmacion}")]}


# ─────────────────────────────────────────────
# 9. ROUTER CONDICIONAL — El Decisor del Flujo
# ─────────────────────────────────────────────
# Analiza el HumanMessage original y verifica si ya se ejecutó
# una herramienta para evitar bucles infinitos.
# Retorna: "ver_menu" | "crear_reserva" | "__end__"
def router_condicional(estado: EstadoRestaurante) -> Literal["ver_menu", "crear_reserva", "__end__"]:
    print("\n" + "·"*60)
    print("  🔀  ROUTER: evaluando destino del flujo...")
    print("·"*60)

    mensajes = estado["messages"]

    # ── CLAVE ANTI-BUCLE: verificar si ya se invocó alguna herramienta ──
    # Si en el historial hay un AIMessage con prefijo [MENU] o [RESERVA],
    # la herramienta ya se ejecutó → finalizar en END.
    herramienta_ejecutada = any(
        isinstance(m, AIMessage) and (
            m.content.startswith("[MENU]") or m.content.startswith("[RESERVA]")
        )
        for m in mensajes
    )

    if herramienta_ejecutada:
        print("  ℹ️  Herramienta ya ejecutada. Ciclo completado.")
        print("  ✅ Decisión: → __end__")
        return "__end__"

    # ── Extraer la intención original del usuario ──
    mensaje_humano = next(
        (m.content for m in mensajes if isinstance(m, HumanMessage)), ""
    )
    texto = mensaje_humano.lower()

    print(f"  🔍 Intención: '{mensaje_humano[:65]}...' "
          if len(mensaje_humano) > 65 else f"  🔍 Intención: '{mensaje_humano}'")

    # Palabras clave para enrutar a ver_menu
    palabras_menu = [
        "menú", "menu", "carta", "platos", "comer", "comida",
        "que tienen", "qué tienen", "tienen", "ofrecen", "opciones",
        "qué hay", "que hay", "ver el menú", "mostrar"
    ]

    # Palabras clave para enrutar a crear_reserva
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

    Estructura:
      START → agent → [router] → ver_menu      → agent → END
                               → crear_reserva → agent → END
                               → END
    """
    print("\n🔧 Construyendo el grafo LangGraph...")

    grafo = StateGraph(EstadoRestaurante)

    # ── Añadir los 3 nodos ──
    grafo.add_node("agent",         nodo_agent)
    grafo.add_node("ver_menu",      nodo_ver_menu)
    grafo.add_node("crear_reserva", nodo_crear_reserva)

    # ── Aristas fijas ──
    grafo.add_edge(START,          "agent")          # Entrada única
    grafo.add_edge("ver_menu",     "agent")          # Ciclo herramienta → agent
    grafo.add_edge("crear_reserva","agent")          # Ciclo herramienta → agent

    # ── Arista condicional (el Router) ──
    # El router decide qué camino tomar DESPUÉS de que el agent responde.
    grafo.add_conditional_edges(
        "agent",
        router_condicional,
        {
            "ver_menu":      "ver_menu",
            "crear_reserva": "crear_reserva",
            "__end__":       END,
        }
    )

    app = grafo.compile()
    print("✅ Grafo compilado exitosamente.\n")
    return app


# ─────────────────────────────────────────────
# 11. FUNCIÓN DE UTILIDAD PARA EJECUTAR EJEMPLOS
# ─────────────────────────────────────────────
def ejecutar_ejemplo(app, numero: int, mensaje_usuario: str) -> None:
    """Ejecuta un ejemplo y muestra la respuesta final del agente."""
    separador = "★" * 60
    print(f"\n\n{separador}")
    print(f"  EJEMPLO {numero}")
    print(f"  Usuario: \"{mensaje_usuario}\"")
    print(f"{separador}")

    estado_inicial = {"messages": [HumanMessage(content=mensaje_usuario)]}
    resultado = app.invoke(estado_inicial)

    mensajes_finales = resultado["messages"]
    respuesta_final  = mensajes_finales[-1].content

    print(f"\n{'─'*60}")
    print(f"  💬 RESPUESTA FINAL AL USUARIO:")
    print(f"{'─'*60}")
    print(respuesta_final)
    print(f"{'─'*60}")
    print(f"  📊 Mensajes en el estado: {len(mensajes_finales)}")


# ─────────────────────────────────────────────
# 12. PUNTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":

    app = construir_grafo()

    # ── EJEMPLO 1: Usuario pide el menú ──
    # Flujo: START → agent → ver_menu → agent → END
    ejecutar_ejemplo(
        app,
        numero=1,
        mensaje_usuario="Hola, ¿me puedes mostrar el menú del día?"
    )

    # ── EJEMPLO 2: Usuario pide una reserva ──
    # Flujo: START → agent → crear_reserva → agent → END
    ejecutar_ejemplo(
        app,
        numero=2,
        mensaje_usuario="Quiero hacer una reserva para tres personas mañana por la noche"
    )

    # ── EJEMPLO 3: Usuario saluda sin pedir herramienta ──
    # Flujo: START → agent → END  (el router elige __end__ directamente)
    ejecutar_ejemplo(
        app,
        numero=3,
        mensaje_usuario="¡Buenas tardes! Muchas gracias por la información."
    )
