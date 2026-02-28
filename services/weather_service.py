import joblib
import numpy as np

model = joblib.load("model/weather_delay_model.pkl")

def predict_delay_probability(weather_data):
    features = np.array([[
        weather_data["temperature"],
        weather_data["wind_speed"],
        weather_data["rain"],
        weather_data["humidity"]
    ]])
    
    probability = model.predict_proba(features)[0][1]
    return float(probability)