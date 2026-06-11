import pandas as pd
import requests
from pathlib import Path
from sklearn.metrics import accuracy_score
import os

# =========================
# 🔑 API KEY
# =========================
API_KEY = os.getenv("GEMINI_API_KEY")

# =========================
# 📁 PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "../data/dataset.csv"
prompt_path = BASE_DIR / "../prompts/zero_shot.txt"

# =========================
# 📊 DATA
# =========================
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

print("Columnas detectadas:", df.columns)

# =========================
# 🧠 PROMPT TEMPLATE
# =========================
with open(prompt_path, "r", encoding="utf-8") as f:
    plantilla = f.read()

# =========================
# 🚀 GEMINI CALL
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

    r = requests.post(url, json=payload)
    data = r.json()

    if "error" in data:
        return "error"

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "error"

# =========================
# 🚀 LOOP + CLEAN LABELS
# =========================
y_true = []
y_pred = []

for _, fila in df.iterrows():
    texto = str(fila["texto"])
    etiqueta_real = str(fila["etiqueta"]).strip().lower()

    prompt = plantilla.replace("{texto}", texto)

    respuesta = call_gemini(prompt)

    # 🔥 limpiar respuesta del modelo
    prediccion = respuesta.strip().lower()

    # opcional: forzar etiquetas válidas
    if "positivo" in prediccion:
        prediccion = "positivo"
    elif "negativo" in prediccion:
        prediccion = "negativo"
    elif "neutro" in prediccion:
        prediccion = "neutro"
    else:
        prediccion = "neutro"

    y_true.append(etiqueta_real)
    y_pred.append(prediccion)

    print("\n======================")
    print("Texto:", texto)
    print("Real:", etiqueta_real)
    print("Pred:", prediccion)

# =========================
# 📊 ACCURACY
# =========================
accuracy = accuracy_score(y_true, y_pred)

print("\n======================")
print("🎯 Accuracy final:", accuracy)