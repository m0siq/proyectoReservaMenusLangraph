"""
=============================================================================
  SISTEMA DE AGENTES PARA RESTAURANTE --- Proyecto LangGraph + OpenAI
=============================================================================
  Arquitectura: StateGraph con PARALELISMO (Send API fan-out / fan-in)

  FLUJO COMPLETO:
  ───────────────────────────────────────────────────────────────────────
  __start__ → agent → [router]
                          │
                          ├─► ver_menu_dispatcher ──┬─► nodo_entrantes   ─┐
                          │    (Send x3 en paralelo) ├─► nodo_principales ─┤
                          │                          └─► nodo_postres      ┘
                          │                                ↓ fan-in
                          │                       ver_menu_aggregator → agent → END
                          │
                          ├─► prereserva_dispatcher ──┬─► nodo_verificar_disp. ─┐
                          │    (Send x2 en paralelo)  └─► nodo_calcular_precio  ┘
                          │                                  ↓ fan-in
                          │                         prereserva_aggregator
                          │                                  ↓
                          │                         crear_reserva  ← ⏸ INTERRUPT (HITL)
                          │                                  ↓
                          │                              agent → END
                          │
                          └─► END

  Human-in-the-Loop (HITL):
    - MemorySaver:      persiste el estado entre turnos (memoria de conversación).
    - interrupt_before: pausa el grafo ANTES de ejecutar crear_reserva para
                        que el usuario apruebe o rechace la reserva.

  Paralelismo con Send API:
    - ver_menu_dispatcher  usa Send() para lanzar 3 nodos de menú en paralelo.
    - prereserva_dispatcher usa Send() para lanzar 2 nodos de verificación en
      paralelo antes de crear la reserva.
    - Los agregadores reciben los resultados parciales (vía operator.add en el
      estado compartido) y los componen en un único mensaje.
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
import operator
from typing import TypedDict, Annotated, Literal, Union

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send                              # ← Send API para paralelismo
from langgraph.checkpoint.memory import MemorySaver
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

    # ── Resultados parciales del fan-out del menú ──
    # operator.add concatena las listas cuando varios nodos paralelos
    # escriben en el mismo campo (cada nodo añade su fragmento).
    secciones_menu: Annotated[list[str], operator.add]

    # ── Resultados del fan-out de pre-reserva ──
    info_prereserva: Annotated[list[str], operator.add]


# ── Sub-estado para el procesamiento paralelo de cada categoría (Map-Reduce) ──
class CategoriaMenuState(TypedDict):
    categoria: str


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
Si recibes [PRE_RESERVA], muestra al cliente la disponibilidad de mesa y el precio
estimado de forma amigable, e indícale que en un momento se le pedirá confirmación
para registrar la reserva.
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


# ═════════════════════════════════════════════
# ███  PARALELISMO 1: MAP-REDUCE DEL MENÚ   ███
# ═════════════════════════════════════════════


# ─────────────────────────────────────────────
# 7b. MAPPER único: procesa una categoría
# ─────────────────────────────────────────────
def nodo_procesar_categoria(estado_seccion: CategoriaMenuState) -> dict:
    """
    Mapper (Fase MAP):
    Procesa una única categoría del menú de forma paralela y
    retorna su sección formateada.
    """
    categoria = estado_seccion["categoria"]
    print(f"\n  🍽️   [MAP] nodo_procesar_categoria ejecutando para '{categoria}'...")
    
    platos = MENU_DEL_DIA.get(categoria, [])
    lineas = [f"{categoria.upper()}:"]
    for p in platos:
        lineas.append(f"  - {p['nombre']}: {p['precio']:.2f}€  [Alérgenos: {p['alergenos']}]")
    
    resultado = "\n".join(lineas)
    print(f"  ✅  [MAP] '{categoria}' completada con éxito.")
    
    # Escribe el fragmento en la lista del estado global (gracias a operator.add)
    return {"secciones_menu": [resultado]}


# ─────────────────────────────────────────────
# 7c. AGGREGATOR del menú (REDUCE)
# ─────────────────────────────────────────────
def nodo_ver_menu_aggregator(estado: EstadoRestaurante) -> dict:
    """
    Aggregator (Fase REDUCE):
    Consolida todos los fragmentos procesados en paralelo
    y genera el mensaje definitivo con el menú integrado.
    """
    print("\n" + "◀"*60)
    print("  🔗  MAP-REDUCE AGGREGATOR (REDUCE): ver_menu — uniendo resultados paralelos")
    print("◀"*60)

    secciones = estado.get("secciones_menu", [])
    print(f"  📦  Secciones reducidas: {len(secciones)} / {len(MENU_DEL_DIA)}")

    fecha = datetime.date.today().strftime("%d/%m/%Y")
    cabecera = f"DATOS DEL MENÚ DEL DÍA ({fecha}):\n"
    cuerpo = "\n\n".join(secciones)
    mensaje_final = cabecera + cuerpo

    print("  ✅  Fase de reducción (Reduce) completada con éxito.")
    return {"messages": [AIMessage(content=f"[MENU]\n{mensaje_final}")]}


# ═════════════════════════════════════════════
# ███  PARALELISMO 2: FAN-OUT PRE-RESERVA  ███
# ═════════════════════════════════════════════


# ─────────────────────────────────────────────
# 8b. NODO paralelo: verificar disponibilidad
# ─────────────────────────────────────────────
def nodo_verificar_disponibilidad(estado: EstadoRestaurante) -> dict:
    """
    Rama paralela: comprueba si hay mesas disponibles (simulado).
    En producción consultaría una BD real de reservas.
    """
    print("\n  🪑  [PARALELO] nodo_verificar_disponibilidad ejecutando...")

    # Extraer número de personas del último mensaje del usuario
    mensaje_usuario = next(
        (m.content for m in reversed(estado["messages"]) if isinstance(m, HumanMessage)),
        ""
    ).lower()

    personas = 2
    for num, palabras in [
        (1, ["una persona", "1 persona"]),
        (2, ["dos personas", "2 personas"]),
        (3, ["tres personas", "3 personas"]),
        (4, ["cuatro personas", "4 personas"]),
        (5, ["cinco personas", "5 personas"]),
        (6, ["seis personas", "6 personas"]),
    ]:
        if any(p in mensaje_usuario for p in palabras):
            personas = num
            break

    # Simulación: mesas disponibles si hay menos de 10 reservas activas
    reservas_activas = len(RESERVAS_REGISTRADAS)
    capacidad_total  = 10
    disponible       = reservas_activas < capacidad_total
    mesas_libres     = max(0, capacidad_total - reservas_activas)

    if disponible:
        resultado = (
            f"✅ DISPONIBILIDAD: Mesa disponible para {personas} personas.\n"
            f"   Mesas libres en el sistema: {mesas_libres}/{capacidad_total}"
        )
    else:
        resultado = (
            f"⚠️  DISPONIBILIDAD: Sin mesas libres para {personas} personas.\n"
            f"   Capacidad completa ({reservas_activas}/{capacidad_total} mesas ocupadas)."
        )

    print(f"  ✅  nodo_verificar_disponibilidad → {'disponible' if disponible else 'sin disponibilidad'}")
    return {"info_prereserva": [resultado]}


# ─────────────────────────────────────────────
# 8c. NODO paralelo: calcular precio estimado
# ─────────────────────────────────────────────
def nodo_calcular_precio(estado: EstadoRestaurante) -> dict:
    """
    Rama paralela: calcula el precio estimado para N personas
    basándose en el menú del día (un plato de cada categoría por persona).
    """
    print("\n  💶  [PARALELO] nodo_calcular_precio ejecutando...")

    mensaje_usuario = next(
        (m.content for m in reversed(estado["messages"]) if isinstance(m, HumanMessage)),
        ""
    ).lower()

    personas = 2
    for num, palabras in [
        (1, ["una persona", "1 persona"]),
        (2, ["dos personas", "2 personas"]),
        (3, ["tres personas", "3 personas"]),
        (4, ["cuatro personas", "4 personas"]),
        (5, ["cinco personas", "5 personas"]),
        (6, ["seis personas", "6 personas"]),
    ]:
        if any(p in mensaje_usuario for p in palabras):
            personas = num
            break

    # Precio medio de cada categoría
    precio_medio_entrante   = sum(p["precio"] for p in MENU_DEL_DIA["entrantes"])   / len(MENU_DEL_DIA["entrantes"])
    precio_medio_principal  = sum(p["precio"] for p in MENU_DEL_DIA["principales"]) / len(MENU_DEL_DIA["principales"])
    precio_medio_postre     = sum(p["precio"] for p in MENU_DEL_DIA["postres"])     / len(MENU_DEL_DIA["postres"])
    precio_medio_por_persona = precio_medio_entrante + precio_medio_principal + precio_medio_postre

    precio_total_estimado = precio_medio_por_persona * personas
    precio_min = (
        min(p["precio"] for p in MENU_DEL_DIA["entrantes"]) +
        min(p["precio"] for p in MENU_DEL_DIA["principales"]) +
        min(p["precio"] for p in MENU_DEL_DIA["postres"])
    ) * personas
    precio_max = (
        max(p["precio"] for p in MENU_DEL_DIA["entrantes"]) +
        max(p["precio"] for p in MENU_DEL_DIA["principales"]) +
        max(p["precio"] for p in MENU_DEL_DIA["postres"])
    ) * personas

    resultado = (
        f"💶 PRECIO ESTIMADO para {personas} persona(s):\n"
        f"   Precio medio por persona: {precio_medio_por_persona:.2f}€\n"
        f"   Total estimado:           {precio_total_estimado:.2f}€\n"
        f"   Rango (mín–máx):         {precio_min:.2f}€ – {precio_max:.2f}€"
    )

    print(f"  ✅  nodo_calcular_precio → {precio_total_estimado:.2f}€ estimado ({personas} personas)")
    return {"info_prereserva": [resultado]}


# ─────────────────────────────────────────────
# 8d. AGGREGATOR pre-reserva — fan-in
# ─────────────────────────────────────────────
def nodo_prereserva_aggregator(estado: EstadoRestaurante) -> dict:
    """
    Aggregator pre-reserva (fan-in).

    Combina la disponibilidad y el precio estimado en un mensaje
    informativo que el agente presentará al usuario antes de confirmar.
    """
    print("\n" + "◀"*60)
    print("  🔗  AGGREGATOR: pre-reserva — combinando 2 resultados paralelos")
    print("◀"*60)

    info = estado.get("info_prereserva", [])
    print(f"  📦  Informes recibidos: {len(info)}")

    cuerpo = "\n\n".join(info)
    mensaje_final = f"[PRE_RESERVA]\n{cuerpo}"

    print("  ✅  Pre-reserva agregada y lista para el agente")
    return {"messages": [AIMessage(content=mensaje_final)]}


# ─────────────────────────────────────────────
# 9. NODO: "crear_reserva"
# ─────────────────────────────────────────────
# ⚠️  Este nodo tiene interrupt_before: el grafo se PAUSA antes de ejecutarlo
#     y espera la aprobación humana desde chat_interactivo.py.
def nodo_crear_reserva(estado: EstadoRestaurante) -> EstadoRestaurante:
    print("\n" + "-"*60)
    print("  📅  NODO: crear_reserva  (aprobado — procesando reserva...)")
    print("-"*60)

    mensajes = estado["messages"]

    # ── Usar el ÚLTIMO HumanMessage para extraer datos ──
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
# 10. ROUTER CONDICIONAL
# ─────────────────────────────────────────────
def router_condicional(
    estado: EstadoRestaurante,
) -> Union[list[Send], str]:
    print("\n" + "·"*60)
    print("  🔀  ROUTER: evaluando destino del flujo...")
    print("·"*60)

    mensajes = estado["messages"]

    # ── CLAVE ANTI-BUCLE con soporte multi-turno ──
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
            or m.content.startswith("[PRE_RESERVA]")
        )
        for m in mensajes_turno_actual
    )

    if herramienta_ejecutada:
        print("  ℹ️  Herramienta ya ejecutada en este turno. Ciclo completado.")
        print("  ✅ Decisión: → __end__")
        return "__end__"

    # ── Extraer la intención del turno actual ──
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
        print("  ✅ Decisión (MAP): → nodo_procesar_categoria (Map-Reduce en paralelo x3)")
        return [
            Send("nodo_procesar_categoria", {"categoria": cat})
            for cat in MENU_DEL_DIA.keys()
        ]
    elif any(p in texto for p in palabras_reserva):
        print("  ✅ Decisión (MAP): → [verificar_disp. ‖ calcular_precio] (paralelo x2)")
        subestado = {"messages": estado["messages"], "info_prereserva": []}
        return [
            Send("nodo_verificar_disponibilidad", subestado),
            Send("nodo_calcular_precio",          subestado),
        ]
    else:
        print("  ✅ Decisión: → __end__  (sin herramienta necesaria)")
        return "__end__"


# ─────────────────────────────────────────────
# 11. CONSTRUCCIÓN DEL GRAFO CON PARALELISMO
# ─────────────────────────────────────────────
def construir_grafo():
    """
    Construye, configura y compila el StateGraph del restaurante
    con MAP-REDUCE dinámico y HITL.

    Estructura completa:
      START → agent ──[router_condicional (MAP)]──► [nodo_procesar_categoria x3] ──► ver_menu_aggregator (REDUCE) ─► agent → END
                          │
                          └───► [nodo_verificar_disp. ‖ nodo_calcular_precio] ──► prereserva_aggregator ──► crear_reserva (HITL) ──► agent → END
    """
    print("\n🔧 Construyendo el grafo LangGraph con MAP-REDUCE + HITL...")

    grafo = StateGraph(EstadoRestaurante)

    # ── Nodos ──
    grafo.add_node("agent",                       nodo_agent)
    
    # Menú
    grafo.add_node("nodo_procesar_categoria",     nodo_procesar_categoria)
    grafo.add_node("ver_menu_aggregator",         nodo_ver_menu_aggregator)

    # Reserva
    grafo.add_node("nodo_verificar_disponibilidad", nodo_verificar_disponibilidad)
    grafo.add_node("nodo_calcular_precio",        nodo_calcular_precio)
    grafo.add_node("prereserva_aggregator",       nodo_prereserva_aggregator)
    grafo.add_node("crear_reserva",               nodo_crear_reserva)

    # ── Edges ──
    grafo.add_edge(START, "agent")

    # ── Edges condicionales desde agent ──
    # router_condicional se encarga de bifurcar en Map-Reduce (retornando Sends)
    # o de terminar (retornando __end__).
    grafo.add_conditional_edges(
        "agent",
        router_condicional,
        {
            "__end__": END,
        }
    )

    # ── Edges de Menú ──
    grafo.add_edge("nodo_procesar_categoria", "ver_menu_aggregator")
    grafo.add_edge("ver_menu_aggregator", "agent")

    # ── Edges de Reserva ──
    grafo.add_edge("nodo_verificar_disponibilidad", "prereserva_aggregator")
    grafo.add_edge("nodo_calcular_precio",          "prereserva_aggregator")
    grafo.add_edge("prereserva_aggregator",         "crear_reserva")
    grafo.add_edge("crear_reserva",                 "agent")

    # ── Checkpointer + HITL ──
    memory = MemorySaver()
    app = grafo.compile(
        checkpointer=memory,
        interrupt_before=["crear_reserva"],
    )

    print("✅ Grafo compilado con MAP-REDUCE dinámico (Send API) + HITL.\n")
    return app
