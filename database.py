import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "/tmp/aoip.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at       TEXT    NOT NULL,
            airline          TEXT,
            terminal         TEXT,
            weather          TEXT,
            day              TEXT,
            hour             INTEGER,
            predicted_delay  REAL,
            delay_probability REAL,
            risk_level       TEXT,
            recommendation   TEXT,
            model_used       TEXT,
            shap_explanation TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS risk_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT    NOT NULL,
            terminal     TEXT,
            hour         INTEGER,
            weather      TEXT,
            risk_score   REAL,
            risk_level   TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS forecast_runs (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at           TEXT    NOT NULL,
            weather              TEXT,
            hours_ahead          INTEGER,
            forecast_avg         REAL,
            forecast_max         REAL,
            peak_warnings_count  INTEGER
        )
    ''')

    conn.commit()
    conn.close()
    print("✓ Database initialised")


# =========================
# PREDICTIONS
# =========================
def save_prediction(airline, terminal, weather, day, hour,
                    predicted_delay, delay_probability, risk_level,
                    recommendation, model_used, shap_explanation=None):
    conn = get_connection()
    conn.execute('''
        INSERT INTO predictions
            (created_at, airline, terminal, weather, day, hour,
             predicted_delay, delay_probability, risk_level,
             recommendation, model_used, shap_explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        airline, terminal, weather, day, hour,
        predicted_delay, delay_probability, risk_level,
        recommendation, model_used,
        json.dumps(shap_explanation) if shap_explanation else None
    ))
    conn.commit()
    conn.close()


def get_predictions(limit=50):
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction_stats():
    conn = get_connection()
    stats = conn.execute('''
        SELECT
            COUNT(*)                                         AS total,
            ROUND(AVG(predicted_delay), 1)                  AS avg_delay,
            ROUND(AVG(delay_probability), 1)                AS avg_probability,
            SUM(CASE WHEN risk_level = "HIGH"   THEN 1 ELSE 0 END) AS high_risk_count,
            SUM(CASE WHEN risk_level = "MEDIUM" THEN 1 ELSE 0 END) AS medium_risk_count,
            SUM(CASE WHEN risk_level = "LOW"    THEN 1 ELSE 0 END) AS low_risk_count
        FROM predictions
    ''').fetchone()
    conn.close()
    return dict(stats) if stats else {}


def get_predictions_by_airline():
    conn = get_connection()
    rows = conn.execute('''
        SELECT airline,
               COUNT(*)                        AS total,
               ROUND(AVG(predicted_delay), 1)  AS avg_delay,
               ROUND(AVG(delay_probability),1) AS avg_probability
        FROM predictions
        GROUP BY airline
        ORDER BY avg_delay DESC
    ''').fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_predictions_by_hour():
    conn = get_connection()
    rows = conn.execute('''
        SELECT hour,
               COUNT(*)                       AS total,
               ROUND(AVG(predicted_delay), 1) AS avg_delay
        FROM predictions
        GROUP BY hour
        ORDER BY hour
    ''').fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =========================
# RISK EVENTS
# =========================
def save_risk_event(terminal, hour, weather, risk_score, risk_level):
    conn = get_connection()
    conn.execute('''
        INSERT INTO risk_events (created_at, terminal, hour, weather, risk_score, risk_level)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), terminal, hour, weather, risk_score, risk_level))
    conn.commit()
    conn.close()


def get_risk_events(limit=50):
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM risk_events ORDER BY created_at DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =========================
# FORECAST RUNS
# =========================
def save_forecast_run(weather, hours_ahead, forecast_avg, forecast_max, peak_warnings_count):
    conn = get_connection()
    conn.execute('''
        INSERT INTO forecast_runs
            (created_at, weather, hours_ahead, forecast_avg, forecast_max, peak_warnings_count)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), weather, hours_ahead,
          forecast_avg, forecast_max, peak_warnings_count))
    conn.commit()
    conn.close()


# Initialise on import
init_db()