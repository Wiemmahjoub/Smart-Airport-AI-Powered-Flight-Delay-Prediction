import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

# ---------- SETTINGS ----------
AIRLINES = ["TunisAir", "Qatar Airways", "Emirates", "Lufthansa", "Air France"]
TERMINALS = ["T1", "T2", "T3"]
WEATHER = ["Clear", "Rain", "Storm"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

NUM_FLIGHTS = 200
NUM_PASSENGER_RECORDS = 1000

# ---------- FLIGHTS DATA ----------
flights = []

for i in range(NUM_FLIGHTS):
    terminal = random.choice(TERMINALS)
    gate = f"G{random.randint(1,10)}"
    dep_time = fake.date_time_this_month()
    arr_time = dep_time + timedelta(hours=random.randint(1,6))
    delay = max(0, int(random.gauss(20, 25)))

    flights.append({
        "flight_id": f"FL{i+1000}",
        "airline": random.choice(AIRLINES),
        "terminal": terminal,
        "gate": gate,
        "departure_time": dep_time,
        "arrival_time": arr_time,
        "day_of_week": random.choice(DAYS),
        "weather": random.choice(WEATHER),
        "delay_minutes": delay
    })

df_flights = pd.DataFrame(flights)
df_flights.to_csv("data/raw/flights.csv", index=False)

# ---------- PASSENGER FLOW ----------
passengers = []

start_time = datetime.now() - timedelta(days=3)

for _ in range(NUM_PASSENGER_RECORDS):
    time = start_time + timedelta(minutes=random.randint(0, 4320))

    passengers.append({
        "timestamp": time,
        "terminal": random.choice(TERMINALS),
        "gate": f"G{random.randint(1,10)}",
        "passenger_count": random.randint(20, 300),
        "security_queue": random.randint(5, 120)
    })

df_passengers = pd.DataFrame(passengers)
df_passengers.to_csv("data/raw/passenger_flow.csv", index=False)

print("✅ Data generated successfully!")
