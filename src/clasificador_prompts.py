import pandas as pd
from pathlib import Path
import requests
import os

# =========================
# 🔑 API KEY (DEBE SER VÁLIDA: AI Studio / AIZA...)
# =========================
API_KEY = os.getenv("GEMINI_API_KEY")

# =========================
# 📁 PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "../data/dataset.csv"
prompt_path = BASE_DIR / "../prompts/zero_shot.txt"

# =========================
# 📊 DATASET
# =========================
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

print("Columnas detectadas:", df.columns)

# =========================
# 🧠 PROMPT
# =========================
with open(prompt_path, "r", encoding="utf-8") as f:
    plantilla = f.read()

# =========================
# 🚀 CALL GEMINI (ROBUSTO)
# =========================
def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(url, json=payload)
    data = response.json()

    # 🔥 Manejo de error real
    if "error" in data:
        return f"API ERROR: {data['error']['message']}"

    # 🔥 Evita crash si no hay respuesta válida
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return f"BAD RESPONSE: {data}"

# =========================
# 🚀 LOOP
# =========================
for _, row in df.iterrows():
    texto = str(row["texto"])
    etiqueta_real = str(row["etiqueta"])

    prompt = plantilla.replace("{texto}", texto)

    prediccion = call_gemini(prompt)

    print("\n======================")
    print("Texto:", texto)
    print("Real:", etiqueta_real)
    print("Predicción:", prediccion)