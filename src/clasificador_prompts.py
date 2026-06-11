import pandas as pd
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

# =========================
# CARGAR .env
# =========================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "No se encontró GEMINI_API_KEY. Verifica tu archivo .env"
    )

print(f"API KEY cargada: {API_KEY[:10]}...")

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "../data/dataset.csv"
prompt_path = BASE_DIR / "../prompts/zero_shot.txt"

# =========================
# DATASET
# =========================
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

print("Columnas detectadas:", list(df.columns))

# =========================
# PROMPT
# =========================
with open(prompt_path, "r", encoding="utf-8") as f:
    plantilla = f.read()

# =========================
# GEMINI
# =========================
def call_gemini(prompt):

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=30
        )

        print("\nSTATUS:", response.status_code)

        data = response.json()

        if "error" in data:
            return f"API ERROR: {data['error']['message']}"

        return (
            data["candidates"][0]
            ["content"]["parts"][0]
            ["text"]
            .strip()
        )

    except Exception as e:
        return f"EXCEPTION: {str(e)}"

# =========================
# LOOP
# =========================
for _, row in df.iterrows():

    texto = str(row["texto"]).strip()
    etiqueta_real = str(row["etiqueta"]).strip()

    prompt = plantilla.replace("{texto}", texto)

    prediccion = call_gemini(prompt)

    print("\n======================")
    print("Texto:", texto)
    print("Real:", etiqueta_real)
    print("Predicción:", prediccion)