import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# =========================
# LOAD DATA & MODEL
# =========================
df = pd.read_csv("data/processed/flights_clean.csv")
df.columns = df.columns.str.lower().str.strip()  # uniformiser les noms

model = joblib.load("model/delay_model.pkl")

# =========================
# DASH APP
# =========================
app = dash.Dash(__name__)
app.title = "Smart Airport – Delay Prediction"

# =========================
# LAYOUT
# =========================
app.layout = html.Div(style={"padding": "30px"}, children=[

    html.H1("✈️ Smart Airport – Flight Delay AI"),

    # -------- INPUTS --------
    html.Div([
        dcc.Dropdown(
            id="airline",
            options=[{"label": a, "value": a} for a in sorted(df["airline"].unique())],
            value=df["airline"].unique()[0],
            placeholder="Select Airline"
        ),

        dcc.Dropdown(
            id="terminal",
            options=[{"label": t, "value": t} for t in sorted(df["terminal"].unique())],
            value=df["terminal"].unique()[0],
            placeholder="Select Terminal"
        ),

        dcc.Dropdown(
            id="day",
            options=[{"label": d, "value": d} for d in df["day_of_week"].unique()],
            value=df["day_of_week"].unique()[0],
            placeholder="Day of Week"
        ),

        dcc.Dropdown(
            id="weather",
            options=[{"label": w, "value": w} for w in df["weather"].unique()],
            value=df["weather"].unique()[0],
            placeholder="Weather"
        ),

        html.Button("Predict Delay", id="predict")
    ], style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "10px"}),

    html.Hr(),

    # -------- OUTPUT --------
    html.H3(id="prediction"),
    dcc.Graph(id="gauge"),

    html.Hr(),

    # -------- CHARTS --------
    html.H3("📊 Delay Distribution"),
    dcc.Graph(id="delay-histogram"),

    html.H3("✈️ Average Delay by Airline"),
    dcc.Graph(id="delay-by-airline")
])

# =========================
# CALLBACK: PREDICTION
# =========================
@app.callback(
    Output("prediction", "children"),
    Output("gauge", "figure"),
    Input("predict", "n_clicks"),
    State("airline", "value"),
    State("terminal", "value"),
    State("day", "value"),
    State("weather", "value"),
    prevent_initial_call=True
)
def predict_delay(n, airline, terminal, day, weather):

    # Créer le DataFrame avec les bonnes colonnes
    X = pd.DataFrame([{
        "airline": airline,
        "terminal": terminal,
        "day_of_week": day,
        "weather": weather
    }])

    try:
        # Si le modèle prédit des minutes de retard (régression)
        pred_minutes = model.predict(X)[0]
        
        # Convertir en probabilité de risque
        if pred_minutes < 15:
            prob = 20
            risk = "LOW"
            color = "green"
        elif pred_minutes < 30:
            prob = 50
            risk = "MEDIUM"
            color = "orange"
        else:
            prob = 85
            risk = "HIGH"
            color = "red"
            
        message = f"⏱️ Predicted delay: {int(pred_minutes)} min | Risk: {risk}"
        
    except:
        # Si le modèle est un classificateur (0/1)
        try:
            prob = model.predict_proba(X)[0][1] * 100
            
            if prob < 40:
                risk = "LOW"
                color = "green"
            elif prob < 70:
                risk = "MEDIUM"
                color = "orange"
            else:
                risk = "HIGH"
                color = "red"
                
            message = f"⏱️ Delay risk: {risk} ({prob:.1f}%)"
            
        except Exception as e:
            print(f"Error: {e}")
            return "❌ Error in prediction", go.Figure()

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={"text": "Delay Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#d4efdf"},
                {"range": [40, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#f5b7b1"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": prob
            }
        }
    ))

    return message, fig

# =========================
# CALLBACK: CHARTS INTERACTIFS
# =========================
@app.callback(
    Output("delay-histogram", "figure"),
    Output("delay-by-airline", "figure"),
    Input("airline", "value"),
    Input("terminal", "value"),
    Input("day", "value"),
    Input("weather", "value")
)
def update_charts(airline, terminal, day, weather):

    filtered_df = df[
        (df["airline"] == airline) &
        (df["terminal"] == terminal) &
        (df["day_of_week"] == day) &
        (df["weather"] == weather)
    ]

    # Si pas de données après filtrage, utiliser toutes les données
    if filtered_df.empty:
        filtered_df = df

    hist_fig = px.histogram(
        filtered_df,
        x="delay_minutes",
        nbins=50,
        title="Departure Delay Histogram",
        color_discrete_sequence=["#3498db"]
    )

    bar_fig = px.bar(
        df.groupby("airline")["delay_minutes"].mean().reset_index().sort_values("delay_minutes", ascending=False),
        x="airline",
        y="delay_minutes",
        title="Average Delay per Airline",
        color_discrete_sequence=["#e74c3c"]
    )

    return hist_fig, bar_fig

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)