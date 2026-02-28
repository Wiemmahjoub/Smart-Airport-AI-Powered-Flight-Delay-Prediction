import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


def generate_hourly_series(df_flights=None):
    """Build an hourly average delay series from real data or synthetic."""
    if df_flights is not None and not df_flights.empty and 'departure_hour' in df_flights.columns:
        try:
            hour_df = df_flights.dropna(subset=['departure_hour']).copy()
            hour_df['departure_hour'] = hour_df['departure_hour'].astype(int)
            series = hour_df.groupby('departure_hour')['delay_minutes'].mean()
            # Fill any missing hours
            full = pd.Series(index=range(24), dtype=float)
            full.update(series)
            full = full.interpolate(method='linear').fillna(method='bfill').fillna(20.0)
            return full.values.tolist()
        except:
            pass

    # Synthetic realistic pattern if no data
    np.random.seed(42)
    base = [
        12, 10, 8, 8, 10, 15,   # 0-5  night/early
        35, 55, 60, 45, 30, 25, # 6-11 morning peak
        20, 18, 20, 25, 30, 55, # 12-17 afternoon build
        65, 60, 45, 35, 25, 15  # 18-23 evening peak then down
    ]
    noise = np.random.normal(0, 3, 24)
    return [max(0, b + n) for b, n in zip(base, noise)]


def forecast_delays(df_flights=None, hours_ahead=12, weather="Clear"):
    """
    Run ARIMA forecast on hourly delay pattern.
    Returns past 24h + forecast for next hours_ahead hours.
    """
    weather_multiplier = {'Clear': 1.0, 'Cloudy': 1.15, 'Rain': 1.40, 'Storm': 1.75}.get(weather, 1.0)

    history = generate_hourly_series(df_flights)

    try:
        model     = ARIMA(history, order=(2, 1, 2))
        fitted    = model.fit()
        forecast  = fitted.forecast(steps=hours_ahead)
        conf_int  = fitted.get_forecast(steps=hours_ahead).conf_int(alpha=0.2)

        forecast_values = [max(0, float(v) * weather_multiplier) for v in forecast]
        lower_bound     = [max(0, float(v) * weather_multiplier) for v in conf_int.iloc[:, 0]]
        upper_bound     = [max(0, float(v) * weather_multiplier) for v in conf_int.iloc[:, 1]]
    except Exception as e:
        # Fallback: simple moving average projection
        avg   = np.mean(history[-6:])
        trend = (history[-1] - history[-6]) / 6
        forecast_values = [max(0, avg + trend * i * weather_multiplier) for i in range(1, hours_ahead + 1)]
        lower_bound     = [max(0, v * 0.75) for v in forecast_values]
        upper_bound     = [v * 1.25 for v in forecast_values]

    # Build time labels
    from datetime import datetime, timedelta
    now          = datetime.now()
    history_times = [(now - timedelta(hours=24 - i)).strftime("%H:%M") for i in range(24)]
    forecast_times = [(now + timedelta(hours=i + 1)).strftime("%H:%M") for i in range(hours_ahead)]

    # Peak hours in forecast
    peak_warnings = []
    for i, (t, v) in enumerate(zip(forecast_times, forecast_values)):
        hour = (now + timedelta(hours=i + 1)).hour
        if v > 45:
            peak_warnings.append({
                "time":    t,
                "delay":   round(v, 1),
                "level":   "CRITICAL" if v > 60 else "HIGH",
                "message": f"Expected {round(v, 1)} min avg delay at {t}"
            })

    return {
        "history": {
            "times":  history_times,
            "values": [round(v, 1) for v in history]
        },
        "forecast": {
            "times":        forecast_times,
            "values":       [round(v, 1) for v in forecast_values],
            "lower_bound":  [round(v, 1) for v in lower_bound],
            "upper_bound":  [round(v, 1) for v in upper_bound]
        },
        "peak_warnings":  peak_warnings,
        "weather_applied": weather,
        "hours_ahead":     hours_ahead
    }


def forecast_weekly(df_flights=None):
    """7-day delay forecast based on day-of-week pattern."""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    if df_flights is not None and not df_flights.empty and 'day_of_week' in df_flights.columns:
        try:
            day_avg = df_flights.groupby('day_of_week')['delay_minutes'].mean()
            values  = [float(day_avg.get(d, 25.0)) for d in days]
        except:
            values = [22, 20, 24, 26, 35, 42, 38]
    else:
        values = [22, 20, 24, 26, 35, 42, 38]

    np.random.seed(99)
    noise  = np.random.normal(0, 2, 7)
    values = [max(0, v + n) for v, n in zip(values, noise)]

    return {
        "days":   days,
        "values": [round(v, 1) for v in values],
        "avg":    round(np.mean(values), 1),
        "peak_day": days[int(np.argmax(values))]
    }