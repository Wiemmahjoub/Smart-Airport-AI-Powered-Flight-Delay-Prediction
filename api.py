import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

from aoip_logger import log_prediction, log_forecast, log_risk, log_error
from database   import (save_prediction, get_predictions, get_prediction_stats,
                         get_predictions_by_airline, get_predictions_by_hour,
                         save_risk_event, get_risk_events,
                         save_forecast_run)

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
    risk_model      = joblib.load("model/risk_model.pkl")
    risk_features   = joblib.load("model/risk_features.pkl")
    risk_le_terminal= joblib.load("model/risk_le_terminal.pkl")
    risk_le_airline = joblib.load("model/risk_le_airline.pkl")
    import json as _json
    with open("model/risk_model_metadata.json") as _f:
        risk_metadata = _json.load(_f)
    print(f"✓ ML Risk model loaded — CV F1: {risk_metadata.get('cv_f1_mean','?')}")
except Exception as e:
    risk_model = risk_features = risk_le_terminal = risk_le_airline = None
    risk_metadata = {}
    print(f"⚠ Risk ML model not found (will use statistical fallback): {e}")

try:
    df_flights = pd.read_csv("Data/processed/flights_clean.csv")
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
    terminal:   str
    hour:       int
    weather:    str
    airline:    str = "TunisAir"
    day_of_week:int = 2

class RiskResponse(BaseModel):
    risk_score:      float
    risk_level:      str
    factors:         dict
    model_used:      str
    shap_explanation:list

class ForecastRequest(BaseModel):
    weather:     str = "Clear"
    hours_ahead: int = 12


# =========================
# HEALTH
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
# PREDICTION
# =========================
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    t_start    = time.time()
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
            log_error("prediction", e, {"airline": req.airline})
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

    duration_ms = round((time.time() - t_start) * 1000, 1)

    # ── Persist to DB & log ───────────────────────────────────
    try:
        save_prediction(
            req.airline, req.terminal, req.weather, req.day, hour,
            round(predicted_delay, 1), round(delay_probability, 1),
            risk_level, recommendation, model_used, shap_explanation
        )
    except Exception as e:
        log_error("db_save_prediction", e)

    log_prediction(
        req.airline, req.terminal, req.weather, req.day, hour,
        round(predicted_delay, 1), round(delay_probability, 1),
        risk_level, model_used, duration_ms
    )

    return PredictionResponse(
        predicted_delay   = round(predicted_delay, 1),
        delay_probability = round(delay_probability, 1),
        risk_level        = risk_level,
        recommendation    = recommendation,
        model_used        = model_used,
        shap_explanation  = shap_explanation
    )


# =========================
# RISK
# =========================
@app.post("/risk", response_model=RiskResponse)
def calculate_risk(req: RiskRequest):
    t_start     = time.time()
    is_peak     = 1 if req.hour in range(7,10) or req.hour in range(17,20) else 0
    is_night    = 1 if req.hour < 6 or req.hour > 22 else 0
    is_weekend  = 1 if req.day_of_week in [5, 6] else 0
    weather_enc = {'Clear':0,'Cloudy':1,'Rain':2,'Storm':3}.get(req.weather, 0)
    flight_len  = 120

    # ── Try ML model first ─────────────────────────────────
    model_used      = "statistical_fallback"
    risk_level      = "MEDIUM"
    score           = 50.0
    shap_explanation= []

    if risk_model is not None:
        try:
            # Encode terminal
            if risk_le_terminal and req.terminal in risk_le_terminal.classes_:
                t_enc = int(risk_le_terminal.transform([req.terminal])[0])
            else:
                t_enc = 0

            # Encode airline
            if risk_le_airline and req.airline in risk_le_airline.classes_:
                a_enc = int(risk_le_airline.transform([req.airline])[0])
            else:
                a_enc = 0

            features_vec = np.array([[
                a_enc, t_enc, weather_enc,
                req.hour, is_peak, is_night,
                is_weekend, req.day_of_week, flight_len
            ]])

            # Predict class and probability
            proba       = risk_model.predict_proba(features_vec)[0]
            classes     = risk_model.classes_
            pred_class  = classes[np.argmax(proba)]
            risk_level  = str(pred_class)

            # Score = weighted probability (LOW=20, MEDIUM=55, HIGH=90)
            w_map = {'LOW': 20, 'MEDIUM': 55, 'HIGH': 90}
            score = sum(w_map.get(c, 50) * p for c, p in zip(classes, proba))
            model_used = f"RandomForest (CV F1={risk_metadata.get('cv_f1_mean','?')})"

            # SHAP explanation
            try:
                import shap
                explainer   = shap.TreeExplainer(risk_model)
                shap_vals   = explainer.shap_values(features_vec)
                feat_names  = risk_features if risk_features else [
                    'airline','terminal','weather','hour',
                    'is_peak','is_night','is_weekend','dow','flight_length'
                ]
                # shap_vals for predicted class
                cls_idx = list(classes).index(risk_level)
                sv      = shap_vals[cls_idx][0] if isinstance(shap_vals, list) else shap_vals[0]
                for name, val in sorted(zip(feat_names, sv), key=lambda x: -abs(x[1])):
                    shap_explanation.append({
                        "feature":   name,
                        "impact":    round(float(val), 3),
                        "direction": "increases risk" if val > 0 else "reduces risk"
                    })
            except Exception as se:
                log_error("shap_risk", se)

        except Exception as me:
            log_error("ml_risk", me)
            # Fallback to statistical
            weather_factor = {'Clear':20,'Cloudy':40,'Rain':65,'Storm':85}.get(req.weather,40)
            peak_factor    = 70 if is_peak else 30
            weekend_factor = 40 if is_weekend else 20
            score          = min(30*0.40 + peak_factor*0.25 + weather_factor*0.25 + weekend_factor*0.10, 100)
            risk_level     = "LOW" if score < 40 else "MEDIUM" if score < 70 else "HIGH"
            model_used     = "statistical_fallback"
    else:
        # Pure statistical fallback
        weather_factor = {'Clear':20,'Cloudy':40,'Rain':65,'Storm':85}.get(req.weather,40)
        peak_factor    = 70 if is_peak else 30
        weekend_factor = 40 if is_weekend else 20
        score          = min(30*0.40 + peak_factor*0.25 + weather_factor*0.25 + weekend_factor*0.10, 100)
        risk_level     = "LOW" if score < 40 else "MEDIUM" if score < 70 else "HIGH"

    duration_ms = round((time.time() - t_start) * 1000, 1)

    try:
        save_risk_event(req.terminal, req.hour, req.weather, round(score,1), risk_level)
    except Exception as e:
        log_error("db_save_risk", e)

    log_risk(req.terminal, req.hour, req.weather, round(score,1), risk_level, duration_ms)

    return RiskResponse(
        risk_score       = round(score, 1),
        risk_level       = risk_level,
        model_used       = model_used,
        shap_explanation = shap_explanation[:6],
        factors          = {
            "weather":    req.weather,
            "hour":       req.hour,
            "is_peak":    bool(is_peak),
            "is_weekend": bool(is_weekend),
            "terminal":   req.terminal,
            "airline":    req.airline,
            "model":      model_used
        }
    )

@app.get("/risk/model-info")
def risk_model_info():
    if risk_metadata:
        return risk_metadata
    return {"status": "ML model not trained yet", "fallback": "statistical"}


# =========================
# FORECAST
# =========================
@app.post("/forecast")
def forecast(req: ForecastRequest):
    t_start = time.time()
    from services.forecasting_service import forecast_delays, forecast_weekly
    hours_ahead = max(1, min(req.hours_ahead, 24))
    hourly  = forecast_delays(df_flights, hours_ahead=hours_ahead, weather=req.weather)
    weekly  = forecast_weekly(df_flights)

    forecast_avg          = round(sum(hourly["forecast"]["values"]) / len(hourly["forecast"]["values"]), 1)
    forecast_max          = round(max(hourly["forecast"]["values"]), 1)
    peak_warnings_count   = len(hourly.get("peak_warnings", []))
    duration_ms           = round((time.time() - t_start) * 1000, 1)

    try:
        save_forecast_run(req.weather, hours_ahead, forecast_avg, forecast_max, peak_warnings_count)
    except Exception as e:
        log_error("db_save_forecast", e)

    log_forecast(req.weather, hours_ahead, forecast_avg, forecast_max, peak_warnings_count, duration_ms)

    return {
        "hourly":       hourly,
        "weekly":       weekly,
        "generated_at": datetime.now().isoformat()
    }


# =========================
# HISTORY ENDPOINTS
# =========================
@app.get("/history/predictions")
def history_predictions(limit: int = 50):
    """Return last N predictions from database."""
    return {"predictions": get_predictions(limit)}

@app.get("/history/predictions/stats")
def prediction_stats():
    """Aggregate stats across all saved predictions."""
    return get_prediction_stats()

@app.get("/history/predictions/by-airline")
def predictions_by_airline():
    return {"data": get_predictions_by_airline()}

@app.get("/history/predictions/by-hour")
def predictions_by_hour():
    return {"data": get_predictions_by_hour()}

@app.get("/history/risk")
def history_risk(limit: int = 50):
    return {"events": get_risk_events(limit)}


# =========================
# INFO
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


# =========================
# SIMULATION
# =========================
class SimulationRequest(BaseModel):
    terminal:     str = "T1"
    weather:      str = "Clear"
    flight_count: int = 10
    start_hour:   int = 8
    end_hour:     int = 10

@app.post("/simulate")
def simulate_scenario(req: SimulationRequest):
    from scenario_simulator import ScenarioSimulator
    sim = ScenarioSimulator()
    result = sim.simulate(
        terminal     = req.terminal,
        weather      = req.weather,
        flight_count = req.flight_count,
        start_hour   = req.start_hour,
        end_hour     = req.end_hour,
        df_flights   = df_flights
    )
    return result