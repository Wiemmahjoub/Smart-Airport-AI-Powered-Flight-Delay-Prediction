import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import httpx
import os
from dash.exceptions import PreventUpdate
from datetime import datetime

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# =========================
# COLOR PALETTE
# =========================
BG_MAIN   = "#0d0f12"
BG_CARD   = "#161a20"
BG_PANEL  = "#1c2028"
BG_NAVBAR = "#0a0c0f"
BG_INPUT  = "#1e2330"
TEXT_PRI  = "#c9cdd6"
TEXT_SEC  = "#6b7280"
TEXT_HEAD = "#e2e4e9"
BORDER    = "#252a35"
ACC_TEAL  = "#4e9e8e"
ACC_BLUE  = "#4a7fa5"
ACC_RED   = "#9b4a4a"
ACC_PURP  = "#6a4e8a"
ACC_AMBER = "#a07a3a"
ACC_GREEN = "#3d7a5c"
ACC_CYAN  = "#2e8b9a"

# =========================
# LOAD DATA
# =========================
print("Loading data and models...")

try:
    df_flights = pd.read_csv("data/processed/flights_clean.csv")
    df_flights.columns = df_flights.columns.str.lower().str.strip()
    # Only parse departure_time if it actually has values
    if 'departure_time' in df_flights.columns and df_flights['departure_time'].notna().any():
        df_flights['departure_time'] = pd.to_datetime(df_flights['departure_time'], errors='coerce')
        df_flights['departure_hour'] = df_flights['departure_time'].dt.hour.astype('Int64')
        df_flights['departure_day']  = df_flights['departure_time'].dt.day_name()
    # Only generate random hours if column truly missing
    if 'departure_hour' not in df_flights.columns:
        np.random.seed(42)
        df_flights['departure_hour'] = np.random.randint(0, 24, size=len(df_flights))
    if 'delay_minutes' in df_flights.columns:
        df_flights['delay_risk'] = df_flights['delay_minutes'].apply(
            lambda d: 'Low' if d <= 15 else 'Medium' if d <= 30 else 'High'
        )
    print(f"✓ Loaded {len(df_flights)} flight records")
except Exception as e:
    print(f"✗ Error loading flights: {e}")
    df_flights = pd.DataFrame()

try:
    df_passengers = pd.read_csv("data/processed/passenger_flow_clean.csv")
    df_passengers.columns = df_passengers.columns.str.lower().str.strip()
    if 'timestamp' in df_passengers.columns and 'hour' not in df_passengers.columns:
        df_passengers['timestamp'] = pd.to_datetime(df_passengers['timestamp'], errors='coerce')
        df_passengers['hour'] = df_passengers['timestamp'].dt.hour
    print("✓ Loaded passenger data")
except:
    df_passengers = pd.DataFrame()
    print("✗ No passenger data, using sample")

try:
    delay_model  = joblib.load("model/delay_model.pkl")
    le_airline   = joblib.load("model/le_airline.pkl")
    le_weather   = joblib.load("model/le_weather.pkl")
    le_terminal  = joblib.load("model/le_terminal.pkl")
    print("✓ Local ML models loaded")
except:
    delay_model = le_airline = le_weather = le_terminal = None
    print("⚠ Local ML models not found")

try:
    weather_model = joblib.load("model/weather_delay_model.pkl")
    print("✓ Weather ML model loaded")
except:
    weather_model = None

try:
    from alerts import AOIPAlertSystem
    alert_system = AOIPAlertSystem()
    print("✓ Alert system loaded")
except:
    alert_system = None
    print("⚠ Alert system not found")


# =========================
# MODULES
# =========================
class PassengerFlowIntelligence:
    def __init__(self):
        self.data = self._load_or_create()

    def _load_or_create(self):
        if not df_passengers.empty and 'hour' in df_passengers.columns:
            return df_passengers
        np.random.seed(42)
        data = []
        for hour in range(24):
            for terminal in ['T1', 'T2', 'T3']:
                for g in range(1, 6):
                    peak = hour in range(6, 10) or hour in range(16, 19)
                    data.append({
                        'hour': hour, 'terminal': terminal,
                        'gate': f"{terminal}-G{g}",
                        'passenger_count': np.random.randint(80, 200) if peak else np.random.randint(10, 60),
                        'congestion_level': np.random.randint(60, 90) if peak else np.random.randint(10, 40),
                        'wait_time': np.random.randint(5, 30)
                    })
        return pd.DataFrame(data)

    def get_heatmap(self, terminal="All"):
        df = self.data.copy()
        if terminal != "All":
            df = df[df['terminal'] == terminal]
        if 'hour' not in df.columns:
            df = self._load_or_create()
            if terminal != "All":
                df = df[df['terminal'] == terminal]
        if 'gate' not in df.columns:
            df['gate'] = 'G1'
        fig = px.density_heatmap(
            df, x='hour', y='gate', z='passenger_count',
            title=f'Passenger Flow Heatmap — {terminal}',
            color_continuous_scale='Cividis',
            labels={'hour': 'Hour of Day', 'gate': 'Gate', 'passenger_count': 'Passengers'}
        )
        fig.update_layout(height=500, paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)
        return fig


class GateOptimizationEngine:
    def __init__(self):
        self.flights = df_flights if not df_flights.empty else self._create_sample()

    def _create_sample(self):
        np.random.seed(42)
        airlines = ['TunisAir', 'Air France', 'Emirates', 'Lufthansa', 'Qatar Airways']
        gates    = [f'T{i}-G{j}' for i in range(1, 4) for j in range(1, 6)]
        base     = {'TunisAir': (45, 120), 'Air France': (20, 60),
                    'Lufthansa': (15, 45), 'Emirates': (10, 35), 'Qatar Airways': (5, 25)}
        data = []
        for i in range(500):
            a = np.random.choice(airlines)
            g = np.random.choice(gates)
            lo, hi = base[a]
            data.append({'flight_id': f'FL{i}', 'airline': a,
                         'terminal': g.split('-')[0], 'gate': g,
                         'delay_minutes': np.random.randint(lo, hi), 'status': 'Delayed'})
        return pd.DataFrame(data)

    def analyze_gate_performance(self):
        gs = self.flights.groupby('gate').agg(
            flight_count=('flight_id', 'count'),
            avg_delay=('delay_minutes', 'mean')
        )
        gs['utilization'] = (gs['flight_count'] / gs['flight_count'].max() * 100).round(1)
        gs['efficiency']  = (100 - gs['avg_delay']).clip(0, 100).round(1)
        return gs.sort_values('efficiency', ascending=False)

    def get_optimization_suggestions(self):
        gs = self.analyze_gate_performance()
        suggestions = []
        worst = gs[gs['efficiency'] < 60].head(2)
        best  = gs[gs['efficiency'] > 85].head(2)
        for idx, (gate, stats) in enumerate(worst.iterrows()):
            if idx < len(best.index):
                suggestions.append({
                    'action':   f'🔀 Reassign flights from {gate} to {best.index[idx]}',
                    'reason':   f'{gate} has {stats["avg_delay"]:.1f} min avg delay',
                    'impact':   'Expected 25–40% delay reduction',
                    'priority': 'HIGH' if stats['efficiency'] < 50 else 'MEDIUM'
                })
        suggestions.extend([
            {'action': '📊 Implement dynamic gate assignment',
             'reason': 'Static assignment causes bottlenecks',
             'impact': 'Improve utilization by 15–20%', 'priority': 'MEDIUM'},
            {'action': '🕒 Stagger schedules at peak gates',
             'reason': 'T1-G1 and T2-G1 overloaded 8–10 AM',
             'impact': 'Reduce congestion by 30%', 'priority': 'HIGH'}
        ])
        return suggestions[:5]

    def get_gate_utilization_chart(self):
        gs = self.analyze_gate_performance()
        fig = go.Figure(data=[
            go.Bar(name='Utilization %', x=gs.index, y=gs['utilization'], marker_color=ACC_BLUE),
            go.Scatter(name='Efficiency %', x=gs.index, y=gs['efficiency'],
                       mode='lines+markers', yaxis='y2', line=dict(color=ACC_TEAL, width=2))
        ])
        fig.update_layout(
            title='Gate Performance Metrics', xaxis_title='Gate', yaxis_title='Utilization (%)',
            yaxis2=dict(title='Efficiency (%)', overlaying='y', side='right', range=[0, 100]),
            height=400, showlegend=True, hovermode='x unified',
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI
        )
        return fig


class OperationalRiskScorer:
    def calculate_risk_score(self, terminal, hour, weather="Clear"):
        if not df_flights.empty:
            t_data = df_flights[df_flights['terminal'] == terminal]
            delay_factor = min(t_data['delay_minutes'].mean() * 2, 100) if not t_data.empty else 30
        else:
            delay_factor = 30
        peak_factor    = 70 if hour in range(7, 10) or hour in range(17, 20) else 30
        weather_factor = {'Clear': 20, 'Cloudy': 40, 'Rain': 65, 'Storm': 85}.get(weather, 40)
        weekend_factor = 40 if datetime.now().weekday() >= 5 else 20
        return min(delay_factor * 0.40 + peak_factor * 0.25 +
                   weather_factor * 0.25 + weekend_factor * 0.10, 100)


passenger_ai = PassengerFlowIntelligence()
gate_ai      = GateOptimizationEngine()
risk_ai      = OperationalRiskScorer()
print("✓ AOIP modules ready")

total_flights = len(df_flights)
avg_delay     = round(df_flights['delay_minutes'].mean(), 1) if not df_flights.empty else 0
high_risk     = len(df_flights[df_flights['delay_risk'] == 'High']) if not df_flights.empty else 0
on_time_pct   = round((df_flights['delay_minutes'] <= 15).mean() * 100, 1) if not df_flights.empty else 0

# =========================
# DASH APP
# =========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "AOIP — Airport Operations Intelligence Platform"

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            * {{ font-family: 'Segoe UI', system-ui, sans-serif; box-sizing: border-box; }}
            body {{ margin: 0; padding: 0; background: {BG_MAIN}; color: {TEXT_PRI}; }}
            ::-webkit-scrollbar {{ width: 6px; }}
            ::-webkit-scrollbar-track {{ background: {BG_MAIN}; }}
            ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
            @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.6; }} }}
            .alert-pulse {{ animation: pulse 2s infinite; }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body>
</html>
'''

PAGE  = {'backgroundColor': BG_MAIN, 'color': TEXT_PRI, 'minHeight': '100vh', 'paddingBottom': '60px'}
CARD  = {'backgroundColor': BG_CARD, 'border': f'1px solid {BORDER}', 'borderRadius': '8px',
         'padding': '28px', 'marginBottom': '24px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.5)'}
PANEL = {'backgroundColor': BG_PANEL, 'border': f'1px solid {BORDER}',
         'borderRadius': '6px', 'padding': '20px'}


def alert_banner(alerts):
    if not alerts:
        return html.Div()
    top   = alerts[0]
    color = {'CRITICAL': '#7a1a1a', 'HIGH': ACC_RED, 'MEDIUM': ACC_AMBER}.get(top['priority'], ACC_BLUE)
    return html.Div(className='alert-pulse', style={
        'backgroundColor': color, 'color': TEXT_HEAD, 'padding': '12px 28px',
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'fontSize': '14px'
    }, children=[
        html.Div([html.Strong(f"⚠ {top['priority']}: {top['title']}  "), html.Span(top['message'])]),
        html.Div(top['details'], style={'color': 'rgba(255,255,255,0.7)', 'fontSize': '12px'})
    ])


def navbar():
    link = {'color': TEXT_SEC, 'margin': '0 14px', 'textDecoration': 'none', 'fontSize': '14px'}
    return html.Div(style={
        'backgroundColor': BG_NAVBAR, 'padding': '14px 28px',
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'borderBottom': f'1px solid {BORDER}'
    }, children=[
        html.Span("AOIP · Airport Operations Intelligence",
                  style={'color': TEXT_HEAD, 'fontSize': '16px', 'fontWeight': '600'}),
        html.Div([
            dcc.Link("Home",           href="/",           style=link),
            dcc.Link("Passenger Flow", href="/passenger",  style=link),
            dcc.Link("Gate Opt.",      href="/gates",      style=link),
            dcc.Link("Risk",           href="/risk",       style=link),
            dcc.Link("Prediction",     href="/prediction", style=link),
            dcc.Link("Forecast",       href="/forecast",   style={**link, 'color': ACC_CYAN}),
            dcc.Link("Analytics",      href="/analytics",  style=link),
        ])
    ])


def kpi_card(label, value, color):
    return html.Div(style={
        **CARD, 'textAlign': 'center', 'flex': '1',
        'minWidth': '160px', 'borderTop': f'3px solid {color}', 'marginBottom': '0'
    }, children=[
        html.Div(str(value), style={'fontSize': '36px', 'color': color, 'fontWeight': '300'}),
        html.Div(label, style={'color': TEXT_SEC, 'fontSize': '13px', 'marginTop': '4px'})
    ])


# =========================
# LAYOUTS
# =========================
def home_layout():
    alerts = alert_system.check_all_alerts(df_flights, passenger_ai.data, "Clear") if alert_system else []
    nav_cards = [
        ("Passenger Flow",    ACC_TEAL,  "/passenger",  "Real-time passenger movement and congestion heatmaps."),
        ("Gate Optimisation", ACC_BLUE,  "/gates",      "Gate utilisation and AI-driven reassignment suggestions."),
        ("Risk Management",   ACC_RED,   "/risk",       "Multi-factor operational risk scoring."),
        ("AI Predictions",    ACC_PURP,  "/prediction", "ML-powered delay forecasting with SHAP explainability."),
        ("Delay Forecast",    ACC_CYAN,  "/forecast",   "Time-series delay forecast for next 24 hours and 7 days."),
        ("Analytics",         ACC_AMBER, "/analytics",  "Weather impact and historical delay analytics."),
    ]
    return html.Div(style=PAGE, children=[
        navbar(),
        alert_banner(alerts),
        dcc.Interval(id='interval-refresh', interval=60*1000, n_intervals=0),
        html.Div(style={'padding': '40px', 'maxWidth': '1200px', 'margin': '0 auto'}, children=[
            html.H1("Airport Operations Intelligence Platform",
                    style={'color': TEXT_HEAD, 'fontWeight': '300', 'textAlign': 'center', 'marginBottom': '6px'}),
            html.P("Monitor, analyse, and optimise airport operations in real-time.",
                   style={'color': TEXT_SEC, 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P(id='last-refresh',
                   style={'color': TEXT_SEC, 'textAlign': 'center', 'fontSize': '12px', 'marginBottom': '30px'}),
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginBottom': '40px'}, children=[
                kpi_card("Total Flights",     total_flights,     ACC_BLUE),
                kpi_card("Avg Delay (min)",   avg_delay,         ACC_AMBER),
                kpi_card("High Risk Flights", high_risk,         ACC_RED),
                kpi_card("On-Time %",         f"{on_time_pct}%", ACC_GREEN),
            ]),
            html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'},
                     children=[
                html.Div(style={
                    **CARD, 'width': '180px', 'textAlign': 'center',
                    'borderTop': f'3px solid {color}', 'marginBottom': '0'
                }, children=[
                    html.H3(label, style={'color': color, 'fontSize': '14px', 'marginTop': '0'}),
                    html.P(desc,   style={'color': TEXT_SEC, 'fontSize': '11px', 'lineHeight': '1.5'}),
                    dcc.Link("Open →", href=href,
                             style={'color': color, 'textDecoration': 'none', 'fontWeight': '600'})
                ]) for label, color, href, desc in nav_cards
            ])
        ])
    ])


def passenger_flow_layout():
    if not df_passengers.empty:
        terminal_list = list(df_passengers['terminal'].unique())
    else:
        terminal_list = list(passenger_ai.data['terminal'].unique())
    opts = [{'label': t, 'value': t} for t in ['All'] + terminal_list]
    return html.Div(style=PAGE, children=[
        navbar(),
        html.Div(style={'padding': '30px 40px', 'maxWidth': '1200px', 'margin': '0 auto'}, children=[
            html.H2("Passenger Flow Monitoring", style={'color': ACC_TEAL, 'fontWeight': '400'}),
            html.P("Interactive heatmaps of passenger distribution across terminals and gates.",
                   style={'color': TEXT_SEC}),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '16px', 'margin': '20px 0'}, children=[
                html.Label("Terminal:", style={'color': TEXT_SEC, 'fontSize': '13px'}),
                dcc.Dropdown(id='terminal-select', options=opts, value='All',
                             style={'width': '180px', 'color': '#000'}),
            ]),
            html.Div(style=CARD, children=[
                dcc.Graph(id='passenger-heatmap', figure=passenger_ai.get_heatmap('All'))
            ])
        ])
    ])


def gate_optimization_layout():
    suggestions = gate_ai.get_optimization_suggestions()
    pri_color   = {'HIGH': ACC_RED, 'MEDIUM': ACC_AMBER, 'LOW': ACC_GREEN}
    return html.Div(style=PAGE, children=[
        navbar(),
        html.Div(style={'padding': '30px 40px', 'maxWidth': '1200px', 'margin': '0 auto'}, children=[
            html.H2("Gate Optimisation", style={'color': ACC_BLUE, 'fontWeight': '400'}),
            html.P("Gate performance metrics and AI-driven reassignment suggestions.", style={'color': TEXT_SEC}),
            html.Div(style=CARD, children=[dcc.Graph(figure=gate_ai.get_gate_utilization_chart())]),
            html.H3("AI Suggestions", style={'color': TEXT_HEAD, 'fontWeight': '400'}),
            html.Div([
                html.Div(style={
                    **PANEL,
                    'borderLeft': f'3px solid {pri_color.get(s["priority"], TEXT_SEC)}',
                    'marginBottom': '12px'
                }, children=[
                    html.Div(s['action'], style={'color': TEXT_HEAD, 'fontWeight': '500'}),
                    html.Div(s['reason'],  style={'color': TEXT_SEC, 'fontSize': '13px', 'marginTop': '4px'}),
                    html.Div(f"Impact: {s['impact']}  ·  Priority: {s['priority']}",
                             style={'color': pri_color.get(s['priority'], TEXT_SEC),
                                    'fontSize': '12px', 'marginTop': '6px'})
                ]) for s in suggestions
            ])
        ])
    ])


def risk_management_layout():
    terminal_opts = (
        [{'label': t, 'value': t} for t in df_flights['terminal'].unique()]
        if not df_flights.empty else [{'label': 'T1', 'value': 'T1'}]
    )
    return html.Div(style=PAGE, children=[
        navbar(),
        html.Div(style={'padding': '30px 40px', 'maxWidth': '800px', 'margin': '0 auto'}, children=[
            html.H2("Operational Risk Management", style={'color': ACC_RED, 'fontWeight': '400'}),
            html.P("Real-time risk scoring based on terminal, time, and weather.", style={'color': TEXT_SEC}),
            html.Div(style=CARD, children=[
                html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'alignItems': 'flex-end'},
                         children=[
                    html.Div([
                        html.Label("Terminal", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                      'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='risk-terminal', options=terminal_opts, value='T1',
                                     style={'width': '120px', 'color': '#000'})
                    ]),
                    html.Div([
                        html.Label("Hour (0–23)", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                         'display': 'block', 'marginBottom': '6px'}),
                        dcc.Input(id='risk-hour', type='number', min=0, max=23,
                                  value=datetime.now().hour,
                                  style={'width': '70px', 'backgroundColor': BG_INPUT,
                                         'color': TEXT_PRI, 'border': f'1px solid {BORDER}',
                                         'borderRadius': '4px', 'padding': '6px'})
                    ]),
                    html.Div([
                        html.Label("Weather", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                     'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='risk-weather',
                                     options=[{'label': w, 'value': w} for w in ['Clear','Cloudy','Rain','Storm']],
                                     value='Clear', style={'width': '140px', 'color': '#000'})
                    ]),
                    html.Button("Calculate Risk", id='calculate-risk-btn', n_clicks=0,
                                style={'backgroundColor': ACC_RED, 'color': TEXT_HEAD, 'border': 'none',
                                       'padding': '10px 22px', 'borderRadius': '5px',
                                       'cursor': 'pointer', 'fontWeight': '600'})
                ])
            ]),
            html.Div(id='risk-calculation-result')
        ])
    ])


def prediction_layout():
    airlines = df_flights['airline'].unique().tolist() if not df_flights.empty else ['TunisAir']
    return html.Div(style=PAGE, children=[
        navbar(),
        html.Div(style={'padding': '30px 40px', 'maxWidth': '1100px', 'margin': '0 auto'}, children=[
            html.H2("AI Delay Prediction", style={'color': ACC_PURP, 'fontWeight': '400'}),
            html.P("ML-powered delay forecasting via FastAPI — with SHAP explainability.",
                   style={'color': TEXT_SEC}),
            html.Div(style=CARD, children=[
                html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'alignItems': 'flex-end'},
                         children=[
                    html.Div([
                        html.Label("Airline", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                     'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='pred-airline',
                                     options=[{'label': a, 'value': a} for a in airlines],
                                     value=airlines[0], style={'width': '200px', 'color': '#000'})
                    ]),
                    html.Div([
                        html.Label("Terminal", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                      'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='pred-terminal',
                                     options=[{'label': t, 'value': t} for t in ['T1', 'T2', 'T3']],
                                     value='T1', style={'width': '100px', 'color': '#000'})
                    ]),
                    html.Div([
                        html.Label("Weather", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                     'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='pred-weather',
                                     options=[{'label': w, 'value': w} for w in ['Clear','Cloudy','Rain','Storm']],
                                     value='Clear', style={'width': '130px', 'color': '#000'})
                    ]),
                    html.Div([
                        html.Label("Day", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                 'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='pred-day',
                                     options=[{'label': d, 'value': d} for d in
                                              ['Monday','Tuesday','Wednesday','Thursday',
                                               'Friday','Saturday','Sunday']],
                                     value=datetime.now().strftime("%A"),
                                     style={'width': '150px', 'color': '#000'})
                    ]),
                    html.Button("Run Prediction", id='predict-btn', n_clicks=0,
                                style={'backgroundColor': ACC_PURP, 'color': TEXT_HEAD, 'border': 'none',
                                       'padding': '10px 22px', 'borderRadius': '5px',
                                       'cursor': 'pointer', 'fontWeight': '600'})
                ])
            ]),
            html.Div(id='prediction-results', style={'marginTop': '20px'}),
            html.Div(style=CARD, children=[
                html.H3("Historical Delay Analysis",
                        style={'color': TEXT_HEAD, 'fontWeight': '400', 'marginTop': '0'}),
                html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
                    html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                        dcc.Graph(figure=(
                            px.bar(
                                df_flights.groupby('airline')['delay_minutes'].mean()
                                          .reset_index().sort_values('delay_minutes', ascending=False),
                                x='airline', y='delay_minutes',
                                title='Average Delay by Airline',
                                color='delay_minutes', color_continuous_scale='Reds'
                            ).update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)
                        ) if not df_flights.empty else go.Figure())
                    ]),
                    html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                        dcc.Graph(figure=(
                            px.histogram(df_flights, x='delay_minutes', nbins=30,
                                         title='Delay Distribution',
                                         color_discrete_sequence=[ACC_BLUE])
                            .update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)
                        ) if not df_flights.empty else go.Figure())
                    ])
                ])
            ])
        ])
    ])


def forecast_layout():
    return html.Div(style=PAGE, children=[
        navbar(),
        html.Div(style={'padding': '30px 40px', 'maxWidth': '1200px', 'margin': '0 auto'}, children=[
            html.H2("Operational Delay Forecast", style={'color': ACC_CYAN, 'fontWeight': '400'}),
            html.P("Time-series delay forecast for the next 24 hours and 7 days ahead.",
                   style={'color': TEXT_SEC}),
            html.Div(style=CARD, children=[
                html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'alignItems': 'flex-end'},
                         children=[
                    html.Div([
                        html.Label("Weather Conditions", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                                'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='forecast-weather',
                                     options=[{'label': w, 'value': w} for w in ['Clear','Cloudy','Rain','Storm']],
                                     value='Clear', style={'width': '160px', 'color': '#000'})
                    ]),
                    html.Div([
                        html.Label("Hours Ahead", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                         'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='forecast-hours',
                                     options=[{'label': f'{h} hours', 'value': h} for h in [6, 12, 18, 24]],
                                     value=12, style={'width': '140px', 'color': '#000'})
                    ]),
                    html.Button("Generate Forecast", id='forecast-btn', n_clicks=0,
                                style={'backgroundColor': ACC_CYAN, 'color': TEXT_HEAD, 'border': 'none',
                                       'padding': '10px 22px', 'borderRadius': '5px',
                                       'cursor': 'pointer', 'fontWeight': '600'})
                ])
            ]),
            html.Div(id='forecast-results')
        ])
    ])


def analytics_layout():
    if not df_flights.empty:
        fig_weather = px.box(
            df_flights, x='weather', y='delay_minutes', color='weather',
            title='Delay Distribution by Weather',
            color_discrete_map={'Clear': ACC_GREEN, 'Cloudy': ACC_BLUE,
                                'Rain': ACC_AMBER, 'Storm': ACC_RED}
        ).update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)

        fig_airline = px.box(
            df_flights, x='airline', y='delay_minutes', color='airline',
            title='Delay Distribution by Airline'
        ).update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)

        hour_df = df_flights.dropna(subset=['departure_hour']).copy()
        hour_df['departure_hour'] = hour_df['departure_hour'].astype(int)
        hour_avg = hour_df.groupby('departure_hour')['delay_minutes'].mean().reset_index()
        fig_hour = px.bar(
            hour_avg, x='departure_hour', y='delay_minutes',
            title='Average Delay by Hour of Day',
            color='delay_minutes', color_continuous_scale='Reds'
        ).update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)

        risk_counts = df_flights['delay_risk'].value_counts().reset_index()
        risk_counts.columns = ['risk', 'count']
        fig_risk = px.pie(
            risk_counts, names='risk', values='count',
            title='Flight Risk Distribution',
            color='risk',
            color_discrete_map={'Low': ACC_GREEN, 'Medium': ACC_AMBER, 'High': ACC_RED}
        ).update_layout(paper_bgcolor=BG_CARD, font_color=TEXT_PRI)

        fig_scatter = px.scatter(
            df_flights.sample(min(500, len(df_flights))),
            x='departure_hour', y='delay_minutes',
            color='weather', size='delay_minutes',
            title='Weather vs Delay Correlation',
            color_discrete_map={'Clear': ACC_GREEN, 'Cloudy': ACC_BLUE,
                                'Rain': ACC_AMBER, 'Storm': ACC_RED}
        ).update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)

        if 'day_of_week' in df_flights.columns:
            day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_avg   = df_flights.groupby('day_of_week')['delay_minutes'].mean().reindex(day_order).reset_index()
            fig_day   = px.bar(
                day_avg, x='day_of_week', y='delay_minutes',
                title='Average Delay by Day of Week',
                color='delay_minutes', color_continuous_scale='Oranges'
            ).update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)
        else:
            fig_day = go.Figure()
    else:
        fig_weather = fig_airline = fig_hour = fig_risk = fig_scatter = fig_day = go.Figure()

    return html.Div(style=PAGE, children=[
        navbar(),
        html.Div(style={'padding': '30px 40px', 'maxWidth': '1200px', 'margin': '0 auto'}, children=[
            html.H2("Advanced Analytics", style={'color': TEXT_HEAD, 'fontWeight': '400'}),
            html.P("Historical weather vs delay correlation and operational insights.",
                   style={'color': TEXT_SEC}),
            html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '400px'}, children=[dcc.Graph(figure=fig_weather)]),
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '400px'}, children=[dcc.Graph(figure=fig_airline)]),
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '400px'}, children=[dcc.Graph(figure=fig_hour)]),
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '400px'}, children=[dcc.Graph(figure=fig_risk)]),
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '400px'}, children=[dcc.Graph(figure=fig_scatter)]),
                html.Div(style={**CARD, 'flex': '1', 'minWidth': '400px'}, children=[dcc.Graph(figure=fig_day)]),
            ])
        ])
    ])


# =========================
# APP LAYOUT
# =========================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


# =========================
# CALLBACKS
# =========================
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/passenger': return passenger_flow_layout()
    if pathname == '/gates':     return gate_optimization_layout()
    if pathname == '/risk':      return risk_management_layout()
    if pathname == '/prediction':return prediction_layout()
    if pathname == '/forecast':  return forecast_layout()
    if pathname == '/analytics': return analytics_layout()
    return home_layout()


@app.callback(Output('last-refresh', 'children'), Input('interval-refresh', 'n_intervals'))
def update_refresh_time(n):
    return f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}  ·  Auto-refresh every 60s"


@app.callback(Output('passenger-heatmap', 'figure'), Input('terminal-select', 'value'))
def update_heatmap(terminal):
    return passenger_ai.get_heatmap(terminal)


@app.callback(
    Output('forecast-results', 'children'),
    Input('forecast-btn', 'n_clicks'),
    State('forecast-weather', 'value'),
    State('forecast-hours', 'value')
)
def run_forecast(n_clicks, weather, hours_ahead):
    if not n_clicks:
        raise PreventUpdate

    try:
        response = httpx.post(f"{API_URL}/forecast", json={
            "weather":     weather,
            "hours_ahead": hours_ahead
        }, timeout=30)
        data = response.json()
    except Exception as ex:
        return html.Div(f"Forecast unavailable: {ex}", style={'color': ACC_RED, 'padding': '20px'})

    hourly   = data["hourly"]
    weekly   = data["weekly"]
    warnings = hourly.get("peak_warnings", [])

    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=hourly["history"]["times"], y=hourly["history"]["values"],
        name="Historical (24h)", line=dict(color=ACC_BLUE, width=2), mode='lines'
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly["forecast"]["times"] + hourly["forecast"]["times"][::-1],
        y=hourly["forecast"]["upper_bound"] + hourly["forecast"]["lower_bound"][::-1],
        fill='toself', fillcolor='rgba(78,158,142,0.15)',
        line=dict(color='rgba(0,0,0,0)'), name="Confidence Interval"
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly["forecast"]["times"], y=hourly["forecast"]["values"],
        name=f"Forecast ({hours_ahead}h)", line=dict(color=ACC_TEAL, width=3, dash='dot'),
        mode='lines+markers', marker=dict(size=6)
    ))
    fig_hourly.update_layout(
        title=f'Delay Forecast — Next {hours_ahead} Hours ({weather} conditions)',
        xaxis_title='Time', yaxis_title='Average Delay (min)',
        height=420, hovermode='x unified',
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI,
        legend=dict(bgcolor=BG_PANEL, bordercolor=BORDER)
    )

    bar_colors = [ACC_RED if v > 40 else ACC_AMBER if v > 25 else ACC_GREEN for v in weekly["values"]]
    fig_weekly = go.Figure(go.Bar(
        x=weekly["days"], y=weekly["values"],
        marker_color=bar_colors,
        text=[f"{v:.0f} min" for v in weekly["values"]],
        textposition='outside'
    ))
    fig_weekly.update_layout(
        title='7-Day Delay Outlook', xaxis_title='Day', yaxis_title='Expected Avg Delay (min)',
        height=320, paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI
    )

    warning_cards = []
    for w in warnings[:4]:
        w_color = ACC_RED if w['level'] == 'CRITICAL' else ACC_AMBER
        warning_cards.append(html.Div(style={
            **PANEL, 'borderLeft': f'3px solid {w_color}', 'marginBottom': '8px',
            'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'
        }, children=[
            html.Span(f"⚠ {w['message']}", style={'color': TEXT_HEAD, 'fontSize': '13px'}),
            html.Span(w['level'], style={'color': w_color, 'fontWeight': '600', 'fontSize': '12px'})
        ]))

    forecast_avg = round(sum(hourly["forecast"]["values"]) / len(hourly["forecast"]["values"]), 1)
    forecast_max = round(max(hourly["forecast"]["values"]), 1)
    peak_day     = weekly.get("peak_day", "—")

    return html.Div([
        html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginBottom': '24px'}, children=[
            html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '150px',
                            'borderTop': f'3px solid {ACC_CYAN}', 'marginBottom': '0'}, children=[
                html.Div(f"{forecast_avg} min", style={'fontSize': '28px', 'color': ACC_CYAN, 'fontWeight': '300'}),
                html.Div("Avg Forecast Delay", style={'color': TEXT_SEC, 'fontSize': '12px'})
            ]),
            html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '150px',
                            'borderTop': f'3px solid {ACC_RED}', 'marginBottom': '0'}, children=[
                html.Div(f"{forecast_max} min", style={'fontSize': '28px', 'color': ACC_RED, 'fontWeight': '300'}),
                html.Div("Peak Forecast Delay", style={'color': TEXT_SEC, 'fontSize': '12px'})
            ]),
            html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '150px',
                            'borderTop': f'3px solid {ACC_AMBER}', 'marginBottom': '0'}, children=[
                html.Div(str(len(warnings)), style={'fontSize': '28px', 'color': ACC_AMBER, 'fontWeight': '300'}),
                html.Div("Peak Hour Warnings", style={'color': TEXT_SEC, 'fontSize': '12px'})
            ]),
            html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '150px',
                            'borderTop': f'3px solid {ACC_PURP}', 'marginBottom': '0'}, children=[
                html.Div(peak_day, style={'fontSize': '28px', 'color': ACC_PURP, 'fontWeight': '300'}),
                html.Div("Busiest Day Ahead", style={'color': TEXT_SEC, 'fontSize': '12px'})
            ]),
        ]),
        html.Div(style=CARD, children=[dcc.Graph(figure=fig_hourly)]),
        html.Div(style=CARD, children=[dcc.Graph(figure=fig_weekly)]),
        html.Div(style=CARD, children=[
            html.H3("⚠ Peak Hour Warnings", style={'color': ACC_AMBER, 'fontWeight': '400', 'marginTop': '0'}),
            html.Div(warning_cards) if warning_cards else
            html.Div("No critical delays forecast in this period.", style={'color': ACC_GREEN, 'fontSize': '14px'})
        ]),
        html.Div(f"Generated at {data['generated_at'][:19].replace('T', ' ')} · Weather: {weather}",
                 style={'color': TEXT_SEC, 'fontSize': '11px', 'textAlign': 'right'})
    ])


@app.callback(
    Output('prediction-results', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('pred-airline', 'value'),
    State('pred-terminal', 'value'),
    State('pred-weather', 'value'),
    State('pred-day', 'value')
)
def predict_delay(n_clicks, airline, terminal, weather, day):
    if not n_clicks:
        raise PreventUpdate

    try:
        response = httpx.post(f"{API_URL}/predict", json={
            "airline": airline, "terminal": terminal,
            "weather": weather, "day": day, "hour": datetime.now().hour
        }, timeout=10)
        data              = response.json()
        predicted_delay   = data["predicted_delay"]
        delay_probability = data["delay_probability"]
        risk              = data["risk_level"]
        rec               = data["recommendation"]
        model_used        = f"FastAPI — {data['model_used']}"
        shap_data         = data["shap_explanation"]
    except Exception as ex:
        hour       = datetime.now().hour
        is_peak    = 1 if hour in range(7, 10) or hour in range(17, 20) else 0
        is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
        a_enc = w_enc = t_enc = 0
        if delay_model and le_airline and le_weather and le_terminal:
            try:
                a_enc = le_airline.transform([airline])[0]   if airline  in le_airline.classes_  else 0
                w_enc = le_weather.transform([weather])[0]   if weather  in le_weather.classes_  else 0
                t_enc = le_terminal.transform([terminal])[0] if terminal in le_terminal.classes_ else 0
                predicted_delay = float(delay_model.predict(
                    np.array([[a_enc, w_enc, t_enc, hour, is_peak, is_weekend]]))[0])
            except:
                predicted_delay = 20.0
        else:
            airline_data    = df_flights[df_flights['airline'] == airline] if not df_flights.empty else pd.DataFrame()
            avg_d           = airline_data['delay_minutes'].mean() if not airline_data.empty else 20
            weather_add     = {'Clear': 0, 'Cloudy': 5, 'Rain': 15, 'Storm': 30}.get(weather, 0)
            predicted_delay = max(0, avg_d * 0.6 + weather_add +
                                 (15 if terminal == "T3" else 5) +
                                 (10 if day in ['Saturday', 'Sunday'] else 0))
        predicted_delay   = max(0, predicted_delay)
        delay_probability = min(100, (predicted_delay / 90) * 100)
        model_used        = f"Local fallback (API error: {ex})"
        shap_data         = []
        if predicted_delay < 15:   risk, rec = "LOW",    "Normal operations"
        elif predicted_delay < 30: risk, rec = "MEDIUM", "Monitor closely"
        else:                      risk, rec = "HIGH",   "Consider schedule adjustment"

    color = ACC_GREEN if risk == "LOW" else ACC_AMBER if risk == "MEDIUM" else ACC_RED

    shap_section = html.Div()
    if shap_data:
        shap_rows = [
            html.Div(style={
                'display': 'flex', 'justifyContent': 'space-between',
                'alignItems': 'center', 'padding': '10px 0',
                'borderBottom': f'1px solid {BORDER}'
            }, children=[
                html.Span(item['feature'], style={'color': TEXT_SEC, 'fontSize': '13px'}),
                html.Span(f"{'▲' if item['impact'] > 0 else '▼'} {abs(item['impact']):.1f} min",
                          style={'color': ACC_RED if item['impact'] > 0 else ACC_GREEN,
                                 'fontWeight': '600', 'fontSize': '14px'}),
                html.Span(item['direction'], style={'color': TEXT_SEC, 'fontSize': '11px'})
            ]) for item in shap_data[:4]
        ]
        shap_section = html.Div(style=CARD, children=[
            html.H3("🔍 Why this prediction?",
                    style={'color': TEXT_HEAD, 'fontWeight': '400', 'marginTop': '0'}),
            html.P("SHAP explainability — top factors contributing to this delay:",
                   style={'color': TEXT_SEC, 'fontSize': '13px', 'marginBottom': '16px'}),
            html.Div(shap_rows)
        ])

    high_alert = html.Div()
    if risk == "HIGH":
        high_alert = html.Div(className='alert-pulse', style={
            'backgroundColor': ACC_RED, 'color': TEXT_HEAD, 'padding': '12px 20px',
            'borderRadius': '6px', 'marginBottom': '16px', 'fontWeight': '600'
        }, children=f"🚨 HIGH DELAY RISK — {airline} on {terminal} — Immediate action recommended")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=predicted_delay,
        title={"text": "Predicted Delay (min)", "font": {"color": TEXT_PRI}},
        number={"font": {"color": color}},
        gauge={'axis': {'range': [0, 120], 'tickcolor': TEXT_SEC},
               'bar': {'color': color}, 'bgcolor': BG_PANEL, 'bordercolor': BORDER,
               'steps': [{'range': [0,  15], 'color': '#1a2e20'},
                         {'range': [15, 30], 'color': '#2e2510'},
                         {'range': [30,120], 'color': '#2e1010'}]}
    ))
    gauge.update_layout(height=280, paper_bgcolor=BG_CARD, font_color=TEXT_PRI)

    prob_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=delay_probability,
        title={"text": "Delay Probability (%)", "font": {"color": TEXT_PRI}},
        number={"suffix": "%", "font": {"color": color}},
        gauge={'axis': {'range': [0, 100], 'tickcolor': TEXT_SEC},
               'bar': {'color': color}, 'bgcolor': BG_PANEL, 'bordercolor': BORDER,
               'steps': [{'range': [0,  30], 'color': '#1a2e20'},
                         {'range': [30, 60], 'color': '#2e2510'},
                         {'range': [60,100], 'color': '#2e1010'}]}
    ))
    prob_gauge.update_layout(height=280, paper_bgcolor=BG_CARD, font_color=TEXT_PRI)

    return html.Div([
        high_alert,
        html.Div(style=CARD, children=[
            html.H3("Prediction Results", style={'color': TEXT_HEAD, 'fontWeight': '400', 'marginTop': '0'}),
            html.Div(f"Model: {model_used}",
                     style={'color': TEXT_SEC, 'fontSize': '12px', 'marginBottom': '16px'}),
            html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={'flex': '1', 'minWidth': '260px'}, children=[dcc.Graph(figure=gauge)]),
                html.Div(style={'flex': '1', 'minWidth': '260px'}, children=[dcc.Graph(figure=prob_gauge)]),
                html.Div(style={**PANEL, 'flex': '1', 'minWidth': '200px'}, children=[
                    html.Div("Risk Level", style={'color': TEXT_SEC, 'fontSize': '12px', 'marginBottom': '4px'}),
                    html.Div(risk, style={'color': color, 'fontSize': '28px',
                                         'fontWeight': '300', 'marginBottom': '12px'}),
                    html.Div("Recommendation", style={'color': TEXT_SEC, 'fontSize': '12px', 'marginBottom': '4px'}),
                    html.Div(rec, style={'color': TEXT_HEAD, 'marginBottom': '16px'}),
                    html.Div("Inputs", style={'color': TEXT_SEC, 'fontSize': '12px', 'marginBottom': '4px'}),
                    html.Ul([
                        html.Li(f"Airline: {airline}",   style={'color': TEXT_PRI}),
                        html.Li(f"Terminal: {terminal}", style={'color': TEXT_PRI}),
                        html.Li(f"Weather: {weather}",   style={'color': TEXT_PRI}),
                        html.Li(f"Day: {day}",           style={'color': TEXT_PRI}),
                    ], style={'margin': '0', 'paddingLeft': '18px'})
                ])
            ])
        ]),
        shap_section
    ])


@app.callback(
    Output('risk-calculation-result', 'children'),
    Input('calculate-risk-btn', 'n_clicks'),
    State('risk-terminal', 'value'),
    State('risk-hour', 'value'),
    State('risk-weather', 'value')
)
def calculate_risk(n_clicks, terminal, hour, weather):
    if not n_clicks:
        raise PreventUpdate

    score = risk_ai.calculate_risk_score(terminal, hour, weather)

    if score < 40:   level, color, icon = "LOW",    ACC_GREEN, "●"
    elif score < 70: level, color, icon = "MEDIUM", ACC_AMBER, "◉"
    else:            level, color, icon = "HIGH",   ACC_RED,   "⬤"

    high_alert = html.Div()
    if score >= 70:
        high_alert = html.Div(className='alert-pulse', style={
            'backgroundColor': ACC_RED, 'color': TEXT_HEAD, 'padding': '12px 20px',
            'borderRadius': '6px', 'marginBottom': '16px', 'fontWeight': '600'
        }, children=f"🚨 HIGH RISK on {terminal} — Score: {score:.0f}/100 — Activate contingency plan")

    extra_alerts = []
    if alert_system:
        alerts = alert_system.check_all_alerts(df_flights, passenger_ai.data, weather)
        extra_alerts = [
            html.Div(style={**PANEL, 'borderLeft': f'3px solid {ACC_RED}', 'marginBottom': '8px'}, children=[
                html.Strong(a['title'], style={'color': TEXT_HEAD}),
                html.Div(a['message'], style={'color': TEXT_SEC, 'fontSize': '13px'})
            ]) for a in alerts[:3]
        ]

    return html.Div([
        high_alert,
        html.Div(style=CARD, children=[
            html.Div(style={'display': 'flex', 'gap': '24px', 'alignItems': 'center', 'flexWrap': 'wrap'},
                     children=[
                html.Div(style={'textAlign': 'center', 'minWidth': '140px'}, children=[
                    html.Div(icon, style={'fontSize': '28px', 'color': color}),
                    html.Div(f"{score:.0f}", style={'fontSize': '56px', 'color': color,
                                                    'fontWeight': '200', 'lineHeight': '1'}),
                    html.Div("Risk Score", style={'color': TEXT_SEC, 'fontSize': '12px', 'marginTop': '4px'})
                ]),
                html.Div(style={**PANEL, 'flex': '1', 'minWidth': '200px'}, children=[
                    html.Table([
                        html.Tr([
                            html.Td(k, style={'color': TEXT_SEC, 'paddingRight': '16px', 'paddingBottom': '8px'}),
                            html.Td(v, style={'color': TEXT_HEAD, 'fontWeight': '500'})
                        ]) for k, v in [("Terminal", terminal), ("Hour", f"{hour}:00"),
                                        ("Weather", weather), ("Level", level)]
                    ])
                ]),
                html.Div(style={**PANEL, 'flex': '2', 'minWidth': '200px'}, children=[
                    html.Div("Recommended Actions",
                             style={'color': TEXT_SEC, 'fontSize': '12px', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li("Increase monitoring frequency",   style={'color': TEXT_PRI, 'marginBottom': '4px'}),
                        html.Li("Prepare backup staff if score > 60", style={'color': TEXT_PRI, 'marginBottom': '4px'}),
                        html.Li("Notify airline operations if score > 75", style={'color': TEXT_PRI, 'marginBottom': '4px'}),
                        html.Li("Activate contingency plan if score > 85", style={'color': TEXT_PRI})
                    ], style={'margin': '0', 'paddingLeft': '18px'})
                ])
            ]),
            html.Div(extra_alerts, style={'marginTop': '20px'}) if extra_alerts else html.Div()
        ])
    ])


# =========================
# RUN
# =========================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 AIRPORT OPERATIONS INTELLIGENCE PLATFORM")
    print("="*50)
    print("📊 Dashboard: http://localhost:8050")
    print("🔌 API:       http://localhost:8000")
    print("📖 API Docs:  http://localhost:8000/docs")
    print("="*50 + "\n")
    app.run_server(host='0.0.0.0', port=8050, debug=False)