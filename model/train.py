import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

df = pd.read_csv("data/Arsenal.csv")
df = df.drop(columns=["Date", "Opponent"])
print(df.head())

y=df.pop("Result")
X= df
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = RandomForestClassifier(
    n_estimators=100,  # número de árboles
    max_depth=None,    # profundidad máxima
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))
dump(clf, "mi_modelo.joblib")
print("Modelo guardado en mi_modelo.joblib")

