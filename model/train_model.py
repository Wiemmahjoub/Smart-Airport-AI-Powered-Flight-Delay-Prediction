import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, roc_auc_score
import joblib

print("Loading data...")
df = pd.read_csv("data/processed/flights_clean.csv")

# ========================
# ENCODE CATEGORICALS
# ========================
le_airline  = LabelEncoder()
le_weather  = LabelEncoder()
le_terminal = LabelEncoder()

df['airline_enc']  = le_airline.fit_transform(df['airline'])
df['weather_enc']  = le_weather.fit_transform(df['weather'])
df['terminal_enc'] = le_terminal.fit_transform(df['terminal'])

# ========================
# MODEL 1: Delay Minutes (Regression)
# ========================
features = ['airline_enc', 'weather_enc', 'terminal_enc',
            'departure_hour', 'is_peak_hour', 'is_weekend']

X = df[features]
y = df['delay_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

delay_model = RandomForestRegressor(n_estimators=100, random_state=42)
delay_model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, delay_model.predict(X_test))
print(f"✓ Delay model trained — MAE: {mae:.1f} minutes")

# ========================
# MODEL 2: Weather Delay Probability (Classification)
# Uses REAL columns from your CSV
# ========================
df['is_delayed'] = (df['delay_minutes'] > 15).astype(int)

weather_features = ['weather_enc', 'departure_hour', 'is_peak_hour', 'is_weekend', 'airline_enc']
X_w = df[weather_features]
y_w = df['is_delayed']

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w, y_w, test_size=0.2, random_state=42)

weather_model = RandomForestClassifier(n_estimators=100, random_state=42)
weather_model.fit(X_train_w, y_train_w)
auc = roc_auc_score(y_test_w, weather_model.predict_proba(X_test_w)[:, 1])
print(f"✓ Weather delay model trained — AUC: {auc:.3f}")

# ========================
# SAVE ALL MODELS
# ========================
import os
os.makedirs("/app/model", exist_ok=True)

joblib.dump(delay_model,   "/app/model/delay_model.pkl")
joblib.dump(weather_model, "/app/model/weather_delay_model.pkl")
joblib.dump(le_airline,    "/app/model/le_airline.pkl")
joblib.dump(le_weather,    "/app/model/le_weather.pkl")
joblib.dump(le_terminal,   "/app/model/le_terminal.pkl")

print("✓ All models saved to /app/model/")
print(f"\nFeature importance (delay model):")
for feat, imp in sorted(zip(features, delay_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")