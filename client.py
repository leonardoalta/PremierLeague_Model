# deploy/client.py
import requests
import json

BASE = "http://127.0.0.1:8000"

# 1) Revisar health
h = requests.get(f"{BASE}/health")
print("HEALTH:", h.status_code, h.text)

# 2) Obtener features requeridas
fs = requests.get(f"{BASE}/features")
if fs.status_code != 200:
    print("No se pudieron obtener las features:", fs.status_code, fs.text)
    exit(1)

fs_json = fs.json()
features = fs_json["features"]
print(f"Features requeridas ({fs_json['count']}):", features)

# 3) Construir un body de ejemplo.
#    ⚠️ Ajusta estos valores según tu caso real. Por defecto, 0.0.
body = {name: 0.0 for name in features}

# Ejemplo: si sabes algunas columnas, puedes sobreescribir:
for k, v in {
    "Is_Home": 1,
    "Goals": 2,
    "Opponent_Goals": 1,
    "Possession": 55,
    "Shots": 12,
    "Shots_on_Target": 6,
    "Fouls": 10,
    "Yellow_Cards": 1,
    "Red_Cards": 0,
    "Corners": 4,
    "Offsides": 2,
    "Pass_Accuracy": 82.5,
    "Season": 2024,
    "Month": 8,
    "Day_of_Week": 6,
    "Last5_Avg_Goals": 1.8,
    "Last5_Win_Rate": 0.6,
}.items():
    if k in body:
        body[k] = v

# 4) Enviar al endpoint /score
resp = requests.post(f"{BASE}/score", json=body)
print("STATUS:", resp.status_code)
print("CONTENT-TYPE:", resp.headers.get("Content-Type"))
try:
    print("JSON:", json.dumps(resp.json(), indent=2))
except Exception:
    print("RESP:", resp.text)

