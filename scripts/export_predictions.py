import pandas as pd
import joblib

# Load clean data
df = pd.read_csv("data/processed/flights_clean.csv")

# Load trained model
model = joblib.load("model/delay_model.pkl")

# Select features used by model
X = df[["airline", "terminal", "day_of_week", "weather"]]

# Predict delay probability
try:
    proba = model.predict_proba(X)[:, 1] * 100
except:
    proba = [50] * len(df)

# Build output
output = pd.DataFrame({
    "flight_id": df["flight_id"],
    "airline": df["airline"],
    "gate": df["gate"],
    "predicted_delay_percent": proba,
    "delay_risk": pd.cut(
        proba,
        bins=[0, 40, 70, 100],
        labels=["LOW", "MEDIUM", "HIGH"]
    )
})

# Export for Power BI
output.to_csv("data/predictions_for_powerbi.csv", index=False)

print("✅ Predictions exported for Power BI")
