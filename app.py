# deploy/app.py
from pathlib import Path
from typing import Dict, Any

import json
import numpy as np
from fastapi import FastAPI, HTTPException, Body  # <-- Body para el ejemplo en /docs
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

app = FastAPI(title="Premier League Match Prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ajusta si quieres restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent           # .../deploy
MODEL_PATH = BASE_DIR / "model" / "mi_modelo.joblib"
FEATURES_PATH = BASE_DIR / "model" / "features.json"

# --- Ejemplo para que Swagger (/docs) muestre el body prellenado ---
EXAMPLE_PAYLOAD = {
    "Is_Home": 1,
    "Goals": 2,
    "Opponent_Goals": 1,
    "Possession": 55,
    "Shots": 12,
    "Shots_On_Target": 6,
    "Passes_Completed": 300,
    "Pass_Accuracy": 82.5,
    "Corners": 4,
    "Crosses": 10,
    "Fouls": 12,
    "Offsides": 2,
    "Opponent_Possession": 45,
    "Opponent_Shots": 8,
    "Opponent_Shots_On_Target": 3,
    "Opponent_Passes_Completed": 280,
    "Opponent_Pass_Accuracy": 78.0,
    "Opponent_Corners": 5,
    "Opponent_Crosses": 7,
    "Opponent_Fouls": 15,
    "Opponent_Offsides": 1,
    "Shot_Efficiency": 0.25,
    "Season": 2024,
    "Month": 8,
    "Day_of_Week": 6,
    "Last5_Avg_Goals": 1.8,
    "Last5_Win_Rate": 0.6
}
# -------------------------------------------------------------------

# Cargar modelo
model = None
try:
    model = load(MODEL_PATH)
    print(f"✅ Modelo cargado: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")

# Cargar orden de features
FEATURE_ORDER = None
try:
    with open(FEATURES_PATH) as f:
        FEATURE_ORDER = json.load(f)
    print(f"✅ Features cargadas ({len(FEATURE_ORDER)}): {FEATURE_ORDER[:5]} ...")
except Exception as e:
    print(f"❌ Error cargando features.json: {e}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "features_loaded": FEATURE_ORDER is not None,
        "n_features": len(FEATURE_ORDER) if FEATURE_ORDER else None,
    }

@app.get("/features")
def features():
    if FEATURE_ORDER is None:
        raise HTTPException(status_code=500, detail="features.json no cargado")
    return {"features": FEATURE_ORDER, "count": len(FEATURE_ORDER)}

@app.post("/score")
def score(payload: Dict[str, Any] = Body(..., example=EXAMPLE_PAYLOAD)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    if FEATURE_ORDER is None:
        raise HTTPException(status_code=500, detail="features.json no cargado")

    # Validar campos faltantes
    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"error": "Faltan campos en el payload", "missing": missing}
        )

    # Construir vector en el orden exacto y convertir a float
    try:
        x = np.array([float(payload[f]) for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error convirtiendo tipos: {e}")

    # Predecir
    try:
        pred = int(model.predict(x)[0])
        proba = float(model.predict_proba(x)[0][-1])  # prob. clase "1" si existe
    except AttributeError:
        pred = int(model.predict(x)[0])
        proba = None

    return {"prediction": pred, "probability": proba}

