# scripts/train_delay_model.py
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/flights_clean.csv")

# =========================
# TARGET
# =========================
y = df["delayed"]

# =========================
# FEATURES (⚠️ minuscules)
# =========================
X = df[["carrier", "terminal", "day_of_week", "weather"]]

# =========================
# PREPROCESSING
# =========================
categorical_features = ["carrier", "terminal", "day_of_week", "weather"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =========================
# MODEL
# =========================
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]
)

# =========================
# TRAIN
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {acc:.2f}")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "model/delay_model.pkl")
print("✅ Model saved to model/delay_model.pkl")
