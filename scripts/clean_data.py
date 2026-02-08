import pandas as pd

# ---------- PASSENGER FLOW ----------
flow = pd.read_csv("data/raw/passenger_flow.csv")

flow["timestamp"] = pd.to_datetime(flow["timestamp"])
flow = flow.dropna()
flow = flow.sort_values("timestamp")

flow.to_csv("data/processed/passenger_flow_clean.csv", index=False)


# ---------- PASSENGERS ----------
passengers = pd.read_csv("data/raw/passengers.csv")

passengers["timestamp"] = pd.to_datetime(passengers["timestamp"])
passengers = passengers.drop_duplicates(subset="passenger_id")

passengers.to_csv("data/processed/passengers_clean.csv", index=False)


# ---------- BAGGAGE ----------
baggage = pd.read_csv("data/raw/baggage.csv")

baggage = baggage.drop_duplicates(subset="baggage_id")

baggage.to_csv("data/processed/baggage_clean.csv", index=False)


print("✅ All data cleaned and saved to data/processed/")
