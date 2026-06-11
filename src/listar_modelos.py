import requests
import os
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ No se encontró GEMINI_API_KEY")
    exit()

print(f"🔑 API KEY: {API_KEY[:10]}...")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

try:
    response = requests.get(url, timeout=30)

    print("\nSTATUS:", response.status_code)
    print("\nRESPUESTA COMPLETA:")
    print(response.text)

    if response.status_code == 200:
        data = response.json()

        print("\n📋 MODELOS DISPONIBLES:")
        print("=" * 50)

        for model in data.get("models", []):
            print(model["name"])

except Exception as e:
    print("❌ ERROR:", e)