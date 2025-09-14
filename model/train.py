# deploy/model/train.py
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

# Rutas robustas relativas a este archivo
BASE_DIR = Path(__file__).resolve().parent              # .../deploy/model
DATA_PATH = BASE_DIR.parent / "data" / "Arsenal.csv"    # .../deploy/data/Arsenal.csv
MODEL_PATH = BASE_DIR / "mi_modelo.joblib"              # .../deploy/model/mi_modelo.joblib
FEATURES_PATH = BASE_DIR / "features.json"              # .../deploy/model/features.json

# 1) Cargar datos
df = pd.read_csv(DATA_PATH)

# 2) Preprocesamiento mínimo (ajusta según tu dataset)
#    Quitamos columnas no numéricas (si no las vas a codificar)
drop_cols = [c for c in ["Date", "Opponent"] if c in df.columns]
df = df.drop(columns=drop_cols)

print(df.head())

# 3) Separar target y features
if "Result" not in df.columns:
    raise ValueError("No se encontró la columna 'Result' en el CSV.")

y = df.pop("Result")
X = df

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Modelo
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# 6) Entrenamiento
clf.fit(X_train, y_train)

# 7) Evaluación
y_pred = clf.predict(X_test)
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# 8) Guardar modelo
dump(clf, MODEL_PATH)
print(f"✅ Modelo guardado en: {MODEL_PATH}")

# 9) Guardar orden exacto de features
with open(FEATURES_PATH, "w") as f:
    json.dump(list(X.columns), f, indent=2)
print(f"✅ Features guardadas en: {FEATURES_PATH}")

