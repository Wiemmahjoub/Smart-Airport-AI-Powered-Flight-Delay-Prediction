import pandas as pd
import os

# Créer les dossiers s'ils n'existent pas
os.makedirs("data/processed", exist_ok=True)

# ---------- LOAD RAW DATA ----------
flights = pd.read_csv("data/raw/flights.csv")

# ---------- DATE PARSING ----------
flights["departure_time"] = pd.to_datetime(flights["departure_time"])
flights["arrival_time"] = pd.to_datetime(flights["arrival_time"])

# ---------- BASIC CLEANING ----------
# Remove negative delays (just in case)
flights["delay_minutes"] = flights["delay_minutes"].clip(lower=0)

# ---------- FEATURE ENGINEERING ----------
# Hour of departure
flights["departure_hour"] = flights["departure_time"].dt.hour

# Peak hours (6–9 AM, 5–9 PM)
flights["is_peak_hour"] = flights["departure_hour"].apply(
    lambda h: 1 if (6 <= h <= 9 or 17 <= h <= 21) else 0
)

# Weekend
flights["is_weekend"] = flights["day_of_week"].isin(["Sat", "Sun"]).astype(int)

# Delay risk label
def delay_risk(delay):
    if delay > 60:
        return "High"
    elif delay > 20:
        return "Medium"
    else:
        return "Low"

flights["delay_risk"] = flights["delay_minutes"].apply(delay_risk)

# ---------- SAVE PROCESSED ----------
flights.to_csv("data/processed/flights_clean.csv", index=False)

print("✅ Flights data cleaned & saved")

# ---------- LOAD PASSENGER DATA ----------
passengers = pd.read_csv("data/raw/passenger_flow.csv")

passengers["timestamp"] = pd.to_datetime(passengers["timestamp"])

# Time features
passengers["hour"] = passengers["timestamp"].dt.hour
passengers["day"] = passengers["timestamp"].dt.day
passengers["weekday"] = passengers["timestamp"].dt.weekday  # 0=Mon

# Congestion level
def congestion_level(count):
    if count > 200:
        return "High"
    elif count > 100:
        return "Medium"
    else:
        return "Low"

passengers["congestion"] = passengers["passenger_count"].apply(congestion_level)

# ---------- SAVE ----------
passengers.to_csv("data/processed/passenger_flow_clean.csv", index=False)

print("✅ Passenger flow cleaned & saved")