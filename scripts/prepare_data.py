import pandas as pd

# Charger le CSV BTS
df = pd.read_csv("data/bts_flights.csv")

# Filtrer seulement tes compagnies
my_airlines = ["Qatar Airways", "TunisAir", "Emirates", "Lufthansa"]
df = df[df["Carrier"].isin(my_airlines)]

# Créer colonnes supplémentaires pour ton app
import numpy as np
np.random.seed(42)

df["Terminal"] = np.random.choice(["T1", "T2", "T3"], size=len(df))
df["Weather"] = np.random.choice(["Clear", "Rain", "Snow", "Fog"], size=len(df))
df["Day_of_Week"] = pd.to_datetime(df["FlightDate"]).dt.day_name()
df["delayed"] = (df["DepDelay"] > 15).astype(int)

# Garder seulement colonnes utiles
df = df[["Carrier", "Origin", "Dest", "CRSDepTime", "DepDelay", "ArrDelay",
         "Terminal", "Weather", "Day_of_Week", "delayed"]]

# Sauvegarder dataset propre
df.to_csv("data/flights_clean.csv", index=False)
print("✅ Dataset prêt pour ton modèle AI")
