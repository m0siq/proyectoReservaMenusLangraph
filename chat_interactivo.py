"""
=============================================================================
  MODO INTERACTIVO — Restaurante La Buena Mesa
=============================================================================
  Escribe tu mensaje y el agente LangGraph responde en tiempo real.
  Escribe 'salir' o 'exit' para terminar.
=============================================================================
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Importar el grafo ya construido desde el módulo principal
from restaurante_agente import construir_grafo
from langchain_core.messages import HumanMessage


def modo_interactivo():
    print("\n" + "="*60)
    print("  🍽️  RESTAURANTE LA BUENA MESA — Asistente Virtual")
    print("="*60)
    print("  Puedes preguntarme sobre el menú o hacer una reserva.")
    print("  Escribe 'salir' para terminar.")
    print("="*60 + "\n")

    # Construir el grafo una sola vez
    app = construir_grafo()

    while True:
        try:
            # Leer input del usuario
            entrada = input("  Tú → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 ¡Hasta pronto!")
            break

        # Condición de salida
        if entrada.lower() in ("salir", "exit", "quit", "adios", "adiós", "bye"):
            print("\n  👋 ¡Hasta pronto! Que disfrute su visita.")
            break

        if not entrada:
            continue

        # Ejecutar el grafo con el mensaje del usuario
        estado_inicial = {"messages": [HumanMessage(content=entrada)]}

        print()  # línea en blanco antes de los logs del grafo

        try:
            resultado = app.invoke(estado_inicial)
            respuesta = resultado["messages"][-1].content

            print(f"\n  🤖 MesaBot →\n")
            # Imprimir la respuesta con sangría para distinguirla
            for linea in respuesta.split("\n"):
                print(f"     {linea}")
            print()

        except Exception as e:
            print(f"\n  ❌ Error al procesar tu mensaje: {e}\n")


if __name__ == "__main__":
    modo_interactivo()
