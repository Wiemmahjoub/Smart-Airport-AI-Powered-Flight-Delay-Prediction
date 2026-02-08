import pandas as pd
import random

# Load passenger flow data
flow = pd.read_csv("data/raw/passenger_flow.csv")

passengers = []
baggage = []

p_id = 1
b_id = 1

for _, row in flow.iterrows():
    count = int(row["passenger_count"])

    for _ in range(count):
        passenger_id = f"P{p_id:05d}"
        baggage_id = f"B{b_id:05d}"

        # Passenger record
        passengers.append({
            "passenger_id": passenger_id,
            "terminal": row["terminal"],
            "gate": row["gate"],
            "timestamp": row["timestamp"]
        })

        # Baggage record
        status = random.choices(
            ["delivered", "delayed", "lost"],
            weights=[85, 10, 5]
        )[0]

        baggage.append({
            "baggage_id": baggage_id,
            "passenger_id": passenger_id,
            "status": status
        })

        p_id += 1
        b_id += 1

# Save generated data
pd.DataFrame(passengers).to_csv("data/raw/passengers.csv", index=False)
pd.DataFrame(baggage).to_csv("data/raw/baggage.csv", index=False)

print("✅ passengers.csv and baggage.csv generated successfully")
