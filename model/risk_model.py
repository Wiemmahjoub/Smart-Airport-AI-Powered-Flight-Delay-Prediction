"""
AOIP — ML Risk Model Trainer
Trains a Random Forest risk classifier (LOW / MEDIUM / HIGH)
from flight data and saves it alongside SHAP explainer.

Run inside the API container:
    python train_risk_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

print("=" * 55)
print("  AOIP — ML Risk Model Training")
print("=" * 55)

# =========================
# LOAD DATA
# =========================
DATA_PATHS = [
    "Data/processed/flights_clean.csv",
    "data/processed/flights_clean.csv",
]

df = None
for p in DATA_PATHS:
    if os.path.exists(p):
        df = pd.read_csv(p)
        df.columns = df.columns.str.lower().str.strip()
        print(f"✓ Loaded {len(df):,} records from {p}")
        break

if df is None:
    print("⚠ No flight CSV found — generating synthetic training data")
    np.random.seed(42)
    n = 8000
    airlines  = ['TunisAir','Air France','Emirates','Lufthansa',
                 'Qatar Airways','Delta','United','Southwest']
    terminals = ['T1','T2','T3']
    weathers  = ['Clear','Cloudy','Rain','Storm']
    w_map     = {'Clear':0,'Cloudy':1,'Rain':2,'Storm':3}

    df = pd.DataFrame({
        'airline':        np.random.choice(airlines,  n),
        'terminal':       np.random.choice(terminals, n),
        'weather':        np.random.choice(weathers,  n, p=[0.45,0.30,0.18,0.07]),
        'departure_hour': np.random.randint(0, 24, n),
        'day_of_week':    np.random.randint(0, 7,  n),
        'delay_minutes':  np.abs(np.random.normal(28, 22, n))
    })

print(f"  Columns: {df.columns.tolist()}")

# =========================
# FEATURE ENGINEERING
# =========================
def build_features(df):
    d = df.copy()

    # Risk label from delay
    if 'delay_minutes' in d.columns:
        d['risk_label'] = pd.cut(
            d['delay_minutes'],
            bins=[-1, 15, 40, 9999],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
    elif 'delay' in d.columns:
        # Binary delay column → derive risk
        d['risk_label'] = d['delay'].map({0: 'LOW', 1: 'HIGH'})
    else:
        raise ValueError("No delay column found")

    # Hour features
    if 'departure_hour' in d.columns:
        d['hour'] = d['departure_hour']
    elif 'time' in d.columns:
        d['hour'] = (d['time'] // 60) % 24
    else:
        d['hour'] = 12

    d['is_peak']    = d['hour'].apply(lambda h: 1 if h in range(7,10) or h in range(17,20) else 0)
    d['is_night']   = d['hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)

    # Day of week
    if 'day_of_week' in d.columns:
        d['dow'] = d['day_of_week']
    elif 'dayofweek' in d.columns:
        d['dow'] = d['dayofweek']
    else:
        d['dow'] = 3
    d['is_weekend'] = d['dow'].apply(lambda x: 1 if x in [5, 6] else 0)

    # Weather encoding
    if 'weather' in d.columns:
        weather_map = {'Clear': 0, 'Cloudy': 1, 'Rain': 2, 'Storm': 3}
        d['weather_enc'] = d['weather'].map(weather_map).fillna(0).astype(int)
    else:
        d['weather_enc'] = 0

    # Terminal encoding
    if 'terminal' in d.columns:
        le_t = LabelEncoder()
        d['terminal_enc'] = le_t.fit_transform(d['terminal'].astype(str))
    else:
        d['terminal_enc'] = 0
        le_t = None

    # Airline encoding
    if 'airline' in d.columns:
        le_a = LabelEncoder()
        d['airline_enc'] = le_a.fit_transform(d['airline'].astype(str))
    elif 'airportfrom' in d.columns:
        le_a = LabelEncoder()
        d['airline_enc'] = le_a.fit_transform(d['airportfrom'].astype(str))
    else:
        d['airline_enc'] = 0
        le_a = None

    # Flight length (if available)
    if 'length' in d.columns:
        d['flight_length'] = d['length'].fillna(d['length'].median())
    else:
        d['flight_length'] = 120

    return d, le_t, le_a

df_feat, le_terminal, le_airline = build_features(df)
df_feat = df_feat.dropna(subset=['risk_label'])

FEATURES = ['airline_enc', 'terminal_enc', 'weather_enc',
            'hour', 'is_peak', 'is_night', 'is_weekend',
            'dow', 'flight_length']

X = df_feat[FEATURES]
y = df_feat['risk_label'].astype(str)

print(f"\n✓ Feature matrix: {X.shape}")
print(f"  Risk distribution:\n{y.value_counts().to_string()}")

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n⏳ Training Random Forest Risk Classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
print(f"✓ CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Test set report
y_pred = model.predict(X_test)
print(f"\n📊 Classification Report:\n{classification_report(y_test, y_pred)}")

# Feature importance
importances = dict(zip(FEATURES, model.feature_importances_))
print("\n🔍 Feature Importances:")
for f, imp in sorted(importances.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {f:<18} {bar} {imp:.3f}")

# =========================
# SAVE MODEL & METADATA
# =========================
os.makedirs("model", exist_ok=True)

joblib.dump(model,       "model/risk_model.pkl")
joblib.dump(FEATURES,    "model/risk_features.pkl")
joblib.dump(le_terminal, "model/risk_le_terminal.pkl")
joblib.dump(le_airline,  "model/risk_le_airline.pkl")

# Save model metadata
import json
metadata = {
    "trained_at":    datetime.now().isoformat(),
    "model_type":    "RandomForestClassifier",
    "n_estimators":  200,
    "features":      FEATURES,
    "classes":       model.classes_.tolist(),
    "cv_f1_mean":    round(float(cv_scores.mean()), 4),
    "cv_f1_std":     round(float(cv_scores.std()),  4),
    "training_rows": int(len(X_train)),
    "test_rows":     int(len(X_test)),
    "feature_importances": {f: round(float(v), 4) for f, v in importances.items()}
}
with open("model/risk_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Saved:")
print("   model/risk_model.pkl")
print("   model/risk_features.pkl")
print("   model/risk_le_terminal.pkl")
print("   model/risk_le_airline.pkl")
print("   model/risk_model_metadata.json")
print(f"\n🎯 Model ready — CV F1: {cv_scores.mean():.3f}")
print("=" * 55)