"""
Script de prueba rápida — verifica que Azure OpenAI responde correctamente.
Ejecutar: python -X utf8 test_conexion.py
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# Cargar .env
load_dotenv()

AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

print("=" * 55)
print("  TEST DE CONEXION — Azure OpenAI")
print("=" * 55)
print(f"  Endpoint  : {AZURE_ENDPOINT}")
print(f"  Deployment: {AZURE_DEPLOYMENT}")
print(f"  API Ver   : {AZURE_API_VER}")
print(f"  API Key   : {AZURE_API_KEY[:8]}...{AZURE_API_KEY[-4:]} ({len(AZURE_API_KEY)} chars)")
print("-" * 55)

# Intentar base sin /api/projects/... (Azure OpenAI clasico)
import re
base_endpoint = re.sub(r"/api/projects.*$", "", AZURE_ENDPOINT.rstrip("/"))
if not base_endpoint.endswith("/"):
    base_endpoint += "/"

print(f"  Endpoint limpio usado: {base_endpoint}")
print("-" * 55)

try:
    print("\n  Enviando mensaje de prueba al modelo...")
    llm = AzureChatOpenAI(
        azure_endpoint=base_endpoint,
        api_key=AZURE_API_KEY,
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VER,
        temperature=0.0,
    )
    respuesta = llm.invoke([HumanMessage(content="Responde solo: CONEXION OK")])
    print(f"\n  RESULTADO: {respuesta.content}")
    print("\n  ✅ EXITO — El modelo responde correctamente.")
    print("  Puedes ejecutar el agente con: python -X utf8 restaurante_agente.py")

except Exception as e:
    print(f"\n  ❌ ERROR: {type(e).__name__}")
    print(f"  Detalle: {e}")
    print("\n  Posibles causas:")
    print("  1. El endpoint no es correcto (prueba sin /api/projects/...)")
    print("  2. La API Key esta mal copiada")
    print("  3. El deployment 'gpt-4o-mini' no coincide exactamente con el nombre en Foundry")
