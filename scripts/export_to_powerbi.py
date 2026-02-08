import pandas as pd
import os

# Créer le dossier pour Power BI
os.makedirs("data/powerbi", exist_ok=True)

# ========== CHARGER LES DONNÉES ==========
flights = pd.read_csv("data/processed/flights_clean.csv")
passengers = pd.read_csv("data/processed/passenger_flow_clean.csv")

# ========== ENRICHIR LES DONNÉES ==========
# Ajouter des colonnes utiles pour Power BI
flights["departure_time"] = pd.to_datetime(flights["departure_time"])
flights["arrival_time"] = pd.to_datetime(flights["arrival_time"])
flights["date"] = flights["departure_time"].dt.date
flights["month"] = flights["departure_time"].dt.month
flights["month_name"] = flights["departure_time"].dt.month_name()
flights["year"] = flights["departure_time"].dt.year

passengers["timestamp"] = pd.to_datetime(passengers["timestamp"])
passengers["date"] = passengers["timestamp"].dt.date
passengers["month"] = passengers["timestamp"].dt.month
passengers["year"] = passengers["timestamp"].dt.year

# ========== CRÉER DES TABLES DE DIMENSION ==========
# Table Dimension: Airlines
dim_airlines = flights[["airline"]].drop_duplicates().reset_index(drop=True)
dim_airlines["airline_id"] = dim_airlines.index + 1

# Table Dimension: Terminals
dim_terminals = flights[["terminal"]].drop_duplicates().reset_index(drop=True)
dim_terminals["terminal_id"] = dim_terminals.index + 1

# Table Dimension: Weather
dim_weather = flights[["weather"]].drop_duplicates().reset_index(drop=True)
dim_weather["weather_id"] = dim_weather.index + 1

# Table Dimension: Dates
date_range = pd.date_range(
    start=flights["departure_time"].min(),
    end=flights["departure_time"].max(),
    freq="D"
)
dim_dates = pd.DataFrame({
    "date": date_range.date,
    "day_name": date_range.day_name(),
    "month": date_range.month,
    "month_name": date_range.month_name(),
    "year": date_range.year,
    "quarter": date_range.quarter,
    "is_weekend": date_range.dayofweek.isin([5, 6]).astype(int)
})

# ========== CRÉER DES TABLES DE FAITS ==========
# Fact Table: Flight Delays
fact_flights = flights[[
    "flight_id", "airline", "terminal", "weather", 
    "day_of_week", "departure_time", "arrival_time", 
    "delay_minutes", "delay_risk", "departure_hour", 
    "is_peak_hour", "is_weekend", "date"
]]

# Fact Table: Passenger Flow
fact_passengers = passengers[[
    "timestamp", "terminal", "gate", "passenger_count", 
    "congestion", "hour", "date"
]]

# ========== CRÉER DES MESURES PRÉCALCULÉES ==========
# Résumé par jour
daily_summary = flights.groupby("date").agg({
    "flight_id": "count",
    "delay_minutes": ["mean", "max", "sum"],
}).reset_index()
daily_summary.columns = ["date", "total_flights", "avg_delay", "max_delay", "total_delay"]

# Résumé par airline
airline_summary = flights.groupby("airline").agg({
    "flight_id": "count",
    "delay_minutes": ["mean", "max"],
}).reset_index()
airline_summary.columns = ["airline", "total_flights", "avg_delay", "max_delay"]
airline_summary["on_time_rate"] = flights.groupby("airline").apply(
    lambda x: (x["delay_minutes"] <= 15).sum() / len(x) * 100
).values

# Résumé par terminal
terminal_summary = flights.groupby("terminal").agg({
    "flight_id": "count",
    "delay_minutes": "mean",
}).reset_index()
terminal_summary.columns = ["terminal", "total_flights", "avg_delay"]

passenger_terminal = passengers.groupby("terminal")["passenger_count"].sum().reset_index()
passenger_terminal.columns = ["terminal", "total_passengers"]

terminal_summary = terminal_summary.merge(passenger_terminal, on="terminal")

# ========== EXPORT VERS EXCEL (POUR POWER BI) ==========
output_file = "data/powerbi/smart_airport_data.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Tables de faits
    fact_flights.to_excel(writer, sheet_name='Fact_Flights', index=False)
    fact_passengers.to_excel(writer, sheet_name='Fact_Passengers', index=False)
    
    # Tables de dimension
    dim_airlines.to_excel(writer, sheet_name='Dim_Airlines', index=False)
    dim_terminals.to_excel(writer, sheet_name='Dim_Terminals', index=False)
    dim_weather.to_excel(writer, sheet_name='Dim_Weather', index=False)
    dim_dates.to_excel(writer, sheet_name='Dim_Dates', index=False)
    
    # Résumés
    daily_summary.to_excel(writer, sheet_name='Summary_Daily', index=False)
    airline_summary.to_excel(writer, sheet_name='Summary_Airline', index=False)
    terminal_summary.to_excel(writer, sheet_name='Summary_Terminal', index=False)

print(f"✅ Données exportées vers: {output_file}")
print("\n📊 Tables créées:")
print("  - Fact_Flights (table de faits principale)")
print("  - Fact_Passengers (flux de passagers)")
print("  - Dim_Airlines, Dim_Terminals, Dim_Weather, Dim_Dates (dimensions)")
print("  - Summary_Daily, Summary_Airline, Summary_Terminal (résumés)")
print("\n🎯 Prochaine étape: Ouvre Power BI Desktop et importe ce fichier Excel")