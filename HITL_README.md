# Human-in-the-Loop — Restaurante La Buena Mesa

## ¿Qué se ha implementado?

Se han añadido dos mecanismos de control humano al grafo LangGraph:

---

## 1. Memoria de conversación (`MemorySaver`)

**Archivo:** `restaurante_agente.py` → función `construir_grafo()`

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = grafo.compile(checkpointer=memory, interrupt_before=["crear_reserva"])
```

- Cada sesión tiene un **`thread_id`** único (UUID generado en `chat_interactivo.py`).
- LangGraph guarda el estado completo (lista de mensajes) en memoria tras cada nodo.
- En el siguiente `invoke(...)` con el mismo `thread_id`, el grafo recupera el checkpoint y **acumula** el nuevo mensaje al historial existente.
- El agente recuerda todo lo que se ha dicho en la sesión.

---

## 2. Aprobación de reservas (`interrupt_before`)

**Archivo:** `restaurante_agente.py` → `construir_grafo()`  
**Archivo:** `chat_interactivo.py` → bucle principal

```python
app = grafo.compile(
    checkpointer=memory,
    interrupt_before=["crear_reserva"],   # ← pausa aquí
)
```

### Flujo completo

```
Usuario escribe "quiero reservar..."
         │
         ▼
  invoke({"messages": [HumanMessage]}, config)
         │
    START → agent → router ──────────────────► ⏸ INTERRUPT
                                                  (antes de crear_reserva)
         │
         ▼  (el grafo devuelve el control)
  chat muestra la respuesta del agente
  chat pregunta: "¿Confirmas la reserva? (s/n)"
         │
    ┌────┴────┐
    │  SÍ    │  NO
    │         │
    ▼         ▼
invoke(None) update_state([RESERVA_CANCELADA],
    │         as_node="crear_reserva")
    │         + invoke(None)
    ▼         │
crear_reserva ▼
    → agent  agent genera respuesta
    → END    de cancelación → END
```

### Código clave en `chat_interactivo.py`

```python
# Detectar interrupt
estado_actual = app.get_state(config)
if "crear_reserva" in (estado_actual.next or []):
    confirmado = pedir_confirmacion_reserva()

    if confirmado:
        resultado_final = app.invoke(None, config)   # reanudar

    else:
        app.update_state(
            config,
            {"messages": [AIMessage(content="[RESERVA_CANCELADA]\n...")]},
            as_node="crear_reserva",  # simula que el nodo ya corrió
        )
        resultado_final = app.invoke(None, config)   # continuar hacia agent
```

---

## 3. Corrección del router multi-turno

Con memoria persistente, el historial acumula mensajes de varios turnos. El check
anti-bucle original miraba **todos** los mensajes y detectaba herramientas de
turnos anteriores, bloqueando el flujo.

**Solución:** solo mirar los mensajes **desde el último `HumanMessage`** (turno actual):

```python
last_human_idx = max(
    (i for i, m in enumerate(mensajes) if isinstance(m, HumanMessage)),
    default=0
)
mensajes_turno_actual = mensajes[last_human_idx:]

herramienta_ejecutada = any(
    isinstance(m, AIMessage) and m.content.startswith(("[MENU]", "[RESERVA]", "[RESERVA_CANCELADA]"))
    for m in mensajes_turno_actual
)
```

---

## Cómo ejecutar

```bash
python chat_interactivo.py
```

### Ejemplo de sesión

```
  Tú → Hola, ¿qué me recomiendas?
  🤖 MesaBot → Buenas, soy MesaBot. Puedo mostrarte el menú del día
               o ayudarte a hacer una reserva. ¿Qué prefieres?

  Tú → Quiero una mesa para 3 personas mañana por la noche
  🤖 MesaBot → Perfecto, reservaré para 3 personas el [fecha] a las 21:00.
               Voy a solicitar tu confirmación antes de registrarla.

  ⏸⏸⏸  APROBACIÓN REQUERIDA — Human in the Loop
  ¿Confirmar reserva? (s = sí / n = no) → s

  ✅ Reserva aprobada. Procesando...
  🤖 MesaBot → ¡Reserva confirmada! Tu código es RES-4821.
               Te esperamos el [fecha] a las 21:00. ¡Hasta pronto!
```
