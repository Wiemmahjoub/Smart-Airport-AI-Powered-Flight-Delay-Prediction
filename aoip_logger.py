import logging
import json
import os
from datetime import datetime

# =========================
# AOIP STRUCTURED LOGGER
# =========================
LOG_DIR  = os.environ.get("LOG_DIR", "/tmp/logs")
LOG_FILE = os.path.join(LOG_DIR, "aoip.log")

os.makedirs(LOG_DIR, exist_ok=True)

# File handler — structured JSON lines
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)

# Console handler — human readable
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

logger = logging.getLogger("aoip")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def log_prediction(airline, terminal, weather, day, hour,
                   predicted_delay, delay_probability, risk_level,
                   model_used, duration_ms=None):
    """Log a prediction request with all inputs and outputs."""
    entry = {
        "event":             "prediction",
        "timestamp":         datetime.now().isoformat(),
        "inputs": {
            "airline":  airline,
            "terminal": terminal,
            "weather":  weather,
            "day":      day,
            "hour":     hour
        },
        "outputs": {
            "predicted_delay":   predicted_delay,
            "delay_probability": delay_probability,
            "risk_level":        risk_level,
            "model_used":        model_used
        },
        "duration_ms": duration_ms
    }
    logger.info(json.dumps(entry))
    return entry


def log_forecast(weather, hours_ahead, forecast_avg, forecast_max,
                 peak_warnings_count, duration_ms=None):
    """Log a forecast request."""
    entry = {
        "event":     "forecast",
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "weather":     weather,
            "hours_ahead": hours_ahead
        },
        "outputs": {
            "forecast_avg":          forecast_avg,
            "forecast_max":          forecast_max,
            "peak_warnings_count":   peak_warnings_count
        },
        "duration_ms": duration_ms
    }
    logger.info(json.dumps(entry))
    return entry


def log_risk(terminal, hour, weather, risk_score, risk_level, duration_ms=None):
    """Log a risk calculation."""
    entry = {
        "event":     "risk",
        "timestamp": datetime.now().isoformat(),
        "inputs":    {"terminal": terminal, "hour": hour, "weather": weather},
        "outputs":   {"risk_score": risk_score, "risk_level": risk_level},
        "duration_ms": duration_ms
    }
    logger.info(json.dumps(entry))
    return entry


def log_error(event, error_msg, context=None):
    """Log an error event."""
    entry = {
        "event":     f"error_{event}",
        "timestamp": datetime.now().isoformat(),
        "error":     str(error_msg),
        "context":   context or {}
    }
    logger.error(json.dumps(entry))
    return entry