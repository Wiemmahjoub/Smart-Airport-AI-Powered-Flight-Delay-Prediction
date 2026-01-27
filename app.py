import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load the flights CSV
flights = pd.read_csv("Data/flights.csv")

# Show the first 5 rows to check
print(flights.head())


# Load data
df = pd.read_csv("data/flights.csv")

# Convert time columns
df["departure_time"] = pd.to_datetime(df["departure_time"])
df["arrival_time"] = pd.to_datetime(df["arrival_time"])

# Initialize app
app = dash.Dash(__name__)

# Charts
status_chart = px.pie(
    df,
    names="status",
    title="Flight Status Distribution"
)

delay_chart = px.bar(
    df,
    x="airline",
    y="delay_minutes",
    title="Delays by Airline",
    text="delay_minutes"
)

# Layout
app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "20px"},
    children=[
        html.H1("✈️ Smart Airport Dashboard", style={"textAlign": "center"}),

        html.H3("Overview"),
        dcc.Graph(figure=status_chart),

        html.H3("Delays"),
        dcc.Graph(figure=delay_chart),
    ]
)

# Run server
if __name__ == "__main__":
    app.run(debug=True)
