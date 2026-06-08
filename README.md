# 🍽️ Sistema de Agentes para Restaurante — LangGraph + Azure OpenAI

Proyecto escolar que implementa un sistema multiagente para un restaurante utilizando **LangGraph** y **GPT-4o-mini** a través de Azure OpenAI.

---

## Arquitectura del Grafo

```
__start__
    │
    ▼
 [agent] ◄─────────────────────┐
    │                          │
    ▼  router_condicional()    │
    ├──► "ver_menu"     ───────┘  (ciclo obligatorio)
    ├──► "crear_reserva" ─────┘  (ciclo obligatorio)
    └──► END
```

| Nodo | Función |
|---|---|
| `agent` | Cerebro del sistema — llama a GPT-4o-mini |
| `ver_menu` | Consulta la BD y devuelve el menú del día |
| `crear_reserva` | Registra una reserva en el sistema |
| `router` | Decide el siguiente nodo según la intención del usuario |

---

## Requisitos

### 1. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 2. Configurar credenciales

Copia el archivo de ejemplo y rellena tus datos de Azure OpenAI:

```powershell
copy .env.example .env
```

Edita `.env` con tus credenciales de **Microsoft Foundry**:

```env
AZURE_OPENAI_ENDPOINT=https://tu-recurso.services.ai.azure.com/
AZURE_OPENAI_API_KEY=tu_api_key_aqui
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-01
OPENAI_TEMPERATURE=0.3
```

> ⚠️ El archivo `.env` está en `.gitignore` y **nunca se sube a GitHub**. Tu API Key está protegida.

### 3. Verificar la conexión (opcional pero recomendado)

```powershell
python -X utf8 .\test_conexion.py
```

Deberías ver:
```
✅ EXITO — El modelo responde correctamente.
```

---

## Modos de ejecución

El proyecto tiene **dos modos** de uso:

---

### Modo 1 — Ejemplos automáticos

Ejecuta 3 casos de uso predefinidos de forma automática y muestra todo el flujo interno del grafo en consola.

**Archivo:** `restaurante_agente.py`

```powershell
python -X utf8 .\restaurante_agente.py
```

**Qué hace cada ejemplo:**

| Ejemplo | Mensaje | Flujo |
|---|---|---|
| 1 | *"Hola, ¿me puedes mostrar el menú del día?"* | `agent → ver_menu → agent → END` |
| 2 | *"Quiero una reserva para tres personas mañana por la noche"* | `agent → crear_reserva → agent → END` |
| 3 | *"¡Buenas tardes! Muchas gracias por la información."* | `agent → END` (sin herramienta) |

**Salida esperada:**

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
  EJEMPLO 1
  Usuario: "Hola, ¿me puedes mostrar el menú del día?"
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

  🧠  NODO: agent  (consultando GPT-4o-mini...)
  🔀  ROUTER: → ver_menu
  🍽️   NODO: ver_menu  (consultando base de datos...)
  🧠  NODO: agent  (consultando GPT-4o-mini...)
  🔀  ROUTER: → __end__

  💬 RESPUESTA FINAL AL USUARIO:
  ¡Aquí tienes el menú del día! ...
```

> **Ideal para:** demostrar en clase el flujo completo del grafo con todos los prints descriptivos.

---

### Modo 2 — Chat interactivo

Permite escribir mensajes manualmente y conversar con el agente en tiempo real.

**Archivo:** `chat_interactivo.py`

```powershell
python -X utf8 .\chat_interactivo.py
```

**Ejemplo de sesión:**

```
============================================================
  🍽️  RESTAURANTE LA BUENA MESA — Asistente Virtual
============================================================
  Puedes preguntarme sobre el menú o hacer una reserva.
  Escribe 'salir' para terminar.
============================================================

  Tú → quiero ver el menú del día

  🤖 MesaBot →
     ¡Aquí tienes el menú del día!
     **Entrantes:** Gazpacho Andaluz - 6.50€ ...

  Tú → quiero reservar una mesa para 2 personas esta noche

  🤖 MesaBot →
     Tu reserva ha sido confirmada.
     Código: RES-4821 | Fecha: 08/06/2026 | Hora: 21:00

  Tú → salir
  👋 ¡Hasta pronto!
```

**Comandos para salir:** `salir`, `exit`, `adios`

> **Ideal para:** probar el sistema libremente y hacer demos en vivo.

---

## Estructura del proyecto

```
proyectoLang/
│
├── restaurante_agente.py   # Agente principal + 3 ejemplos automáticos
├── chat_interactivo.py     # Modo chat manual
├── test_conexion.py        # Verificación de credenciales Azure
│
├── .env                    # 🔒 Credenciales (NO se sube a GitHub)
├── .env.example            # Plantilla de credenciales
├── .gitignore              # Protege .env
└── requirements.txt        # Dependencias del proyecto
```

---

## Dependencias

| Paquete | Versión | Para qué sirve |
|---|---|---|
| `langgraph` | ≥ 0.2.0 | Construcción del grafo de agentes |
| `langchain-core` | ≥ 0.3.0 | Mensajes y abstracciones base |
| `langchain-openai` | ≥ 0.2.0 | Conexión con Azure OpenAI |
| `python-dotenv` | ≥ 1.0.0 | Carga de variables desde `.env` |
| `openai` | ≥ 1.30.0 | SDK de OpenAI |

---

## Conceptos clave del código

| Concepto | Descripción |
|---|---|
| `TypedDict` + `add_messages` | Define el estado compartido que viaja entre nodos |
| `StateGraph` | Contenedor del grafo cíclico |
| `add_edge()` | Arista fija (siempre se toma ese camino) |
| `add_conditional_edges()` | Arista dinámica decidida por el router |
| `AzureChatOpenAI` | Instancia del LLM real conectado a tu deployment |
| `SystemMessage` | Prompt de sistema que define la personalidad del agente |
