"""
=============================================================================
  MODO INTERACTIVO — Restaurante La Buena Mesa
  (con Paralelismo Send API + Human-in-the-Loop)
=============================================================================
  Funcionalidades activas:

  1. PARALELISMO — Send API (fan-out / fan-in)
     a) Menú: ver_menu_dispatcher lanza 3 nodos en paralelo
        (entrantes, principales, postres) → ver_menu_aggregator
     b) Reserva: prereserva_dispatcher lanza 2 nodos en paralelo
        (verificar_disponibilidad, calcular_precio) → prereserva_aggregator

  2. MEMORIA DE CONVERSACIÓN
     Cada sesión tiene un thread_id único. LangGraph guarda el estado
     completo (historial de mensajes) en el MemorySaver entre turnos, por
     lo que el agente recuerda el contexto de toda la sesión.

  3. APROBACIÓN DE RESERVAS (Human-in-the-Loop)
     Antes de ejecutar nodo_crear_reserva, el grafo hace INTERRUPT y
     devuelve el control aquí. El usuario decide si confirmar o rechazar.

     ┌─ Flujo menú (paralelo) ────────────────────────────────────────────┐
     │  usuario → agent → dispatcher → [entrantes ‖ principales ‖ postres]│
     │                    aggregator → agent → respuesta al usuario        │
     └────────────────────────────────────────────────────────────────────┘

     ┌─ Flujo reserva (paralelo + HITL) ──────────────────────────────────┐
     │  usuario → agent → dispatcher → [verificar_disp. ‖ calc_precio]    │
     │                    aggregator → ⏸ INTERRUPT (aprobación humana)     │
     │  SI confirma  → invoke(None) → crear_reserva → agent → END         │
     │  SI rechaza   → update_state → invoke(None)  → agent → END         │
     └────────────────────────────────────────────────────────────────────┘

  Escribe 'salir' o 'exit' para terminar.
=============================================================================
"""
import sys
import io
import uuid
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from restaurante_agente import construir_grafo
from langchain_core.messages import HumanMessage, AIMessage


# ─────────────────────────────────────────────
# Helpers de presentación
# ─────────────────────────────────────────────
SEPARADOR = "="*60
LINEA     = "─"*60

def mostrar_respuesta(contenido: str, prefijo: str = "  🤖 MesaBot →") -> None:
    """Imprime la respuesta del agente con formato, omitiendo prefijos internos."""
    # Los mensajes [MENU], [RESERVA], [RESERVA_CANCELADA] son internos del grafo;
    # el agente los procesa y genera su propia respuesta humana. Si por algún
    # motivo llegaran aquí, los filtramos.
    for tag in ("[MENU]", "[RESERVA]", "[RESERVA_CANCELADA]", "[PRE_RESERVA]"):
        if contenido.startswith(tag):
            return
    print(f"\n{prefijo}\n")
    for linea in contenido.split("\n"):
        print(f"     {linea}")
    print()


def pedir_confirmacion_reserva() -> bool:
    """
    Muestra el aviso de aprobación HITL y devuelve True si el usuario confirma.
    Este punto es el 'checkpoint humano': el grafo está pausado esperando aquí.
    """
    print(f"\n  {'⏸ '*20}")
    print("  ⚠️   APROBACIÓN REQUERIDA — Human in the Loop")
    print("  El agente quiere registrar una reserva en el sistema.")
    print("  ¿Confirmas la operación?")
    print(f"  {'⏸ '*20}")

    while True:
        respuesta = input("  ¿Confirmar reserva? (s = sí / n = no) → ").strip().lower()
        if respuesta in ("s", "si", "sí", "yes", "y"):
            return True
        if respuesta in ("n", "no"):
            return False
        print("  Por favor escribe 's' para confirmar o 'n' para cancelar.")


# ─────────────────────────────────────────────
# Bucle principal
# ─────────────────────────────────────────────
def modo_interactivo():
    print(f"\n{SEPARADOR}")
    print("  🍽️  RESTAURANTE LA BUENA MESA — Asistente Virtual")
    print(SEPARADOR)
    print("  Puedes preguntarme sobre el menú o hacer una reserva.")
    print("  Escribe 'salir' para terminar.")
    print(SEPARADOR + "\n")

    # ── Construir el grafo (ya incluye MemorySaver + interrupt_before) ──
    app = construir_grafo()

    # ── thread_id único por sesión: identifica este hilo de conversación ──
    # LangGraph usa este ID para guardar y recuperar el estado del checkpoint.
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}
    print(f"  🔑 Sesión iniciada  │  thread_id: {thread_id[:8]}...\n")

    while True:
        # ── Leer input del usuario ──
        try:
            entrada = input("  Tú → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 ¡Hasta pronto!")
            break

        if entrada.lower() in ("salir", "exit", "quit", "adios", "adiós", "bye"):
            print("\n  👋 ¡Hasta pronto! Que disfrute su visita.")
            break

        if not entrada:
            continue

        print()  # línea en blanco antes de los logs del grafo

        # ══════════════════════════════════════════════════════════════
        # INVOCACIÓN 1: enviar el nuevo mensaje del usuario.
        # LangGraph añade el HumanMessage al checkpoint existente
        # (gracias a add_messages + MemorySaver) y ejecuta hasta el
        # primer interrupt o hasta END.
        # ══════════════════════════════════════════════════════════════
        try:
            resultado = app.invoke(
                {"messages": [HumanMessage(content=entrada)]},
                config
            )
        except Exception as e:
            print(f"\n  ❌ Error al procesar tu mensaje: {e}\n")
            continue

        # ── Mostrar la respuesta del agente tras la primera invocación ──
        if resultado and resultado.get("messages"):
            mostrar_respuesta(resultado["messages"][-1].content)

        # ══════════════════════════════════════════════════════════════
        # COMPROBACIÓN HITL: ¿el grafo está pausado antes de crear_reserva?
        # app.get_state(config).next devuelve los nodos que se ejecutarán
        # a continuación. Si contiene "crear_reserva", estamos en INTERRUPT.
        # ══════════════════════════════════════════════════════════════
        estado_actual = app.get_state(config)
        if "crear_reserva" in (estado_actual.next or []):

            # ── Mostrar el checkpoint humano ──
            confirmado = pedir_confirmacion_reserva()

            if confirmado:
                # ── APROBAR: reanudar el grafo desde el checkpoint ──
                # invoke(None, config) continúa desde donde se pausó,
                # es decir, ejecuta nodo_crear_reserva y sigue hasta END.
                print("\n  ✅ Reserva aprobada. Procesando...")
                try:
                    resultado_final = app.invoke(None, config)
                    if resultado_final and resultado_final.get("messages"):
                        mostrar_respuesta(resultado_final["messages"][-1].content)
                except Exception as e:
                    print(f"\n  ❌ Error al confirmar la reserva: {e}\n")

            else:
                # ── RECHAZAR: inyectar mensaje de cancelación y reanudar ──
                # update_state con as_node="crear_reserva" hace que LangGraph
                # trate este update como si nodo_crear_reserva hubiera retornado
                # el mensaje [RESERVA_CANCELADA]. El flujo continúa hacia "agent",
                # que genera una respuesta amigable de cancelación.
                print("\n  ❌ Reserva rechazada. Informando al agente...")
                app.update_state(
                    config,
                    {
                        "messages": [
                            AIMessage(
                                content=(
                                    "[RESERVA_CANCELADA]\n"
                                    "El cliente ha rechazado confirmar la reserva. "
                                    "No se ha registrado ninguna reserva en el sistema."
                                )
                            )
                        ]
                    },
                    as_node="crear_reserva",  # simula que crear_reserva ya corrió
                )
                try:
                    resultado_final = app.invoke(None, config)
                    if resultado_final and resultado_final.get("messages"):
                        mostrar_respuesta(resultado_final["messages"][-1].content)
                except Exception as e:
                    print(f"\n  ❌ Error al cancelar la reserva: {e}\n")


if __name__ == "__main__":
    modo_interactivo()
