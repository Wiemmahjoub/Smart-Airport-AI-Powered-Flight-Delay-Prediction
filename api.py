from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = FastAPI(
    title="AOIP ML API",
    description="Airport Operations Intelligence Platform — ML Inference API",
    version="1.0.0"
)

# =========================
# LOAD MODELS & DATA
# =========================
try:
    delay_model  = joblib.load("model/delay_model.pkl")
    le_airline   = joblib.load("model/le_airline.pkl")
    le_weather   = joblib.load("model/le_weather.pkl")
    le_terminal  = joblib.load("model/le_terminal.pkl")
    print("✓ Delay model loaded")
except Exception as e:
    delay_model = le_airline = le_weather = le_terminal = None
    print(f"⚠ Delay model not loaded: {e}")

try:
    weather_model = joblib.load("model/weather_delay_model.pkl")
    print("✓ Weather model loaded")
except Exception as e:
    weather_model = None
    print(f"⚠ Weather model not loaded: {e}")

try:
    df_flights = pd.read_csv("data/processed/flights_clean.csv")
    df_flights.columns = df_flights.columns.str.lower().str.strip()
    print(f"✓ Loaded {len(df_flights)} flight records for forecasting")
except:
    df_flights = None
    print("⚠ No flight data for forecasting, using synthetic series")


# =========================
# REQUEST / RESPONSE MODELS
# =========================
class PredictionRequest(BaseModel):
    airline:  str
    terminal: str
    weather:  str
    day:      str
    hour:     int = None

class PredictionResponse(BaseModel):
    predicted_delay:   float
    delay_probability: float
    risk_level:        str
    recommendation:    str
    model_used:        str
    shap_explanation:  list

class RiskRequest(BaseModel):
    terminal: str
    hour:     int
    weather:  str

class RiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    factors:    dict

class ForecastRequest(BaseModel):
    weather:     str = "Clear"
    hours_ahead: int = 12


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {
        "service": "AOIP ML API",
        "version": "1.0.0",
        "status":  "running",
        "models": {
            "delay_model":   delay_model   is not None,
            "weather_model": weather_model is not None
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    hour       = req.hour if req.hour is not None else datetime.now().hour
    is_peak    = 1 if hour in range(7, 10) or hour in range(17, 20) else 0
    is_weekend = 1 if req.day in ['Saturday', 'Sunday'] else 0
    a_enc = w_enc = t_enc = 0

    predicted_delay = 20.0
    model_used      = "statistical"

    if delay_model and le_airline and le_weather and le_terminal:
        try:
            a_enc = int(le_airline.transform([req.airline])[0])   if req.airline   in le_airline.classes_   else 0
            w_enc = int(le_weather.transform([req.weather])[0])   if req.weather   in le_weather.classes_   else 0
            t_enc = int(le_terminal.transform([req.terminal])[0]) if req.terminal  in le_terminal.classes_  else 0
            features        = np.array([[a_enc, w_enc, t_enc, hour, is_peak, is_weekend]])
            predicted_delay = float(delay_model.predict(features)[0])
            model_used      = "random_forest"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model error: {e}")

    predicted_delay   = max(0, predicted_delay)
    delay_probability = min(100, (predicted_delay / 90) * 100)

    if weather_model:
        try:
            w_feat            = np.array([[w_enc, hour, is_peak, is_weekend, a_enc]])
            delay_probability = float(weather_model.predict_proba(w_feat)[0][1]) * 100
        except:
            pass

    if predicted_delay < 15:
        risk_level     = "LOW"
        recommendation = "Normal operations"
    elif predicted_delay < 30:
        risk_level     = "MEDIUM"
        recommendation = "Monitor closely, prepare for minor delays"
    else:
        risk_level     = "HIGH"
        recommendation = "Consider schedule adjustment, notify passengers"

    shap_explanation = []
    if delay_model:
        try:
            import shap
            explainer   = shap.TreeExplainer(delay_model)
            input_data  = np.array([[a_enc, w_enc, t_enc, hour, is_peak, is_weekend]])
            shap_values = explainer.shap_values(input_data)[0]
            feat_names  = ['airline', 'weather', 'terminal', 'hour', 'peak_hour', 'weekend']
            for name, val in sorted(zip(feat_names, shap_values), key=lambda x: -abs(x[1])):
                shap_explanation.append({
                    "feature":   name,
                    "impact":    round(float(val), 2),
                    "direction": "increases delay" if val > 0 else "reduces delay"
                })
        except:
            pass

    return PredictionResponse(
        predicted_delay   = round(predicted_delay, 1),
        delay_probability = round(delay_probability, 1),
        risk_level        = risk_level,
        recommendation    = recommendation,
        model_used        = model_used,
        shap_explanation  = shap_explanation
    )


# =========================
# RISK ENDPOINT
# =========================
@app.post("/risk", response_model=RiskResponse)
def calculate_risk(req: RiskRequest):
    weather_factor = {'Clear': 20, 'Cloudy': 40, 'Rain': 65, 'Storm': 85}.get(req.weather, 40)
    peak_factor    = 70 if req.hour in range(7, 10) or req.hour in range(17, 20) else 30
    weekend_factor = 40 if datetime.now().weekday() >= 5 else 20
    delay_factor   = 30

    score = min(
        delay_factor   * 0.40 +
        peak_factor    * 0.25 +
        weather_factor * 0.25 +
        weekend_factor * 0.10,
        100
    )

    if score < 40:   risk_level = "LOW"
    elif score < 70: risk_level = "MEDIUM"
    else:            risk_level = "HIGH"

    return RiskResponse(
        risk_score = round(score, 1),
        risk_level = risk_level,
        factors    = {
            "weather_factor": weather_factor,
            "peak_factor":    peak_factor,
            "weekend_factor": weekend_factor,
            "delay_factor":   delay_factor
        }
    )


# =========================
# FORECAST ENDPOINT
# =========================
@app.post("/forecast")
def forecast(req: ForecastRequest):
    from services.forecasting_service import forecast_delays, forecast_weekly
    hours_ahead = max(1, min(req.hours_ahead, 24))
    hourly = forecast_delays(df_flights, hours_ahead=hours_ahead, weather=req.weather)
    weekly = forecast_weekly(df_flights)
    return {
        "hourly":       hourly,
        "weekly":       weekly,
        "generated_at": datetime.now().isoformat()
    }


# =========================
# INFO ENDPOINTS
# =========================
@app.get("/airlines")
def get_airlines():
    if le_airline is not None:
        return {"airlines": le_airline.classes_.tolist()}
    return {"airlines": ["TunisAir", "Air France", "Emirates", "Lufthansa", "Qatar Airways"]}

@app.get("/weather-options")
def get_weather():
    return {"options": ["Clear", "Cloudy", "Rain", "Storm"]}

@app.get("/model-info")
def model_info():
    info = {"models_loaded": {}}
    if delay_model:
        info["models_loaded"]["delay_model"] = {
            "type":         "RandomForestRegressor",
            "n_estimators": delay_model.n_estimators,
            "features":     ['airline', 'weather', 'terminal', 'hour', 'peak_hour', 'weekend']
        }
    if weather_model:
        info["models_loaded"]["weather_model"] = {
            "type":         "RandomForestClassifier",
            "n_estimators": weather_model.n_estimators,
            "classes":      weather_model.classes_.tolist()
        }
    return info