import dash
from dash import html, dcc, Input, Output, State, dash_table
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
# Airport theme — deep purple night sky
BG_MAIN   = "#08050f"
BG_CARD   = "#100a1e"
BG_PANEL  = "#160e28"
BG_NAVBAR = "#060310"
BG_INPUT  = "#130c22"
TEXT_PRI  = "#d0c8e8"
TEXT_SEC  = "#6a5a8a"
TEXT_HEAD = "#ece8f8"
BORDER    = "#2a1a48"
ACC_TEAL  = "#a070f0"
ACC_BLUE  = "#8050d0"
ACC_RED   = "#e050a0"
ACC_PURP  = "#c080ff"
ACC_AMBER = "#9060e0"
ACC_GREEN = "#70a0f0"
ACC_CYAN  = "#d090ff"

# =========================
# LOAD DATA
# =========================
print("Loading data and models...")

try:
    df_flights = pd.read_csv("data/processed/flights_clean.csv")
    df_flights.columns = df_flights.columns.str.lower().str.strip()
    if 'departure_time' in df_flights.columns and df_flights['departure_time'].notna().any():
        df_flights['departure_time'] = pd.to_datetime(df_flights['departure_time'], errors='coerce')
        df_flights['departure_hour'] = df_flights['departure_time'].dt.hour.astype('Int64')
        df_flights['departure_day']  = df_flights['departure_time'].dt.day_name()
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
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

            body {{
                font-family: 'Inter', sans-serif;
                background: {BG_MAIN};
                color: {TEXT_PRI};
                overflow-x: hidden;
            }}

            /* ── Animated sky background ── */
            body::before {{
                content: '';
                position: fixed;
                inset: 0;
                z-index: 0;
                background:
                    radial-gradient(ellipse 80% 60% at 15% 10%, rgba(120,40,200,0.22) 0%, transparent 60%),
                    radial-gradient(ellipse 60% 50% at 85% 20%, rgba(80,20,160,0.18) 0%, transparent 55%),
                    radial-gradient(ellipse 50% 40% at 50% 90%, rgba(160,60,220,0.12) 0%, transparent 50%),
                    linear-gradient(170deg, #08050f 0%, #0e0818 40%, #08050f 100%);
                pointer-events: none;
            }}

            /* ── Star field ── */
            body::after {{
                content: '';
                position: fixed;
                inset: 0;
                z-index: 0;
                background-image:
                    radial-gradient(1px 1px at 10% 15%, rgba(220,200,255,0.7) 0%, transparent 100%),
                    radial-gradient(1px 1px at 25% 40%, rgba(200,180,255,0.5) 0%, transparent 100%),
                    radial-gradient(1px 1px at 40% 8%,  rgba(230,210,255,0.6) 0%, transparent 100%),
                    radial-gradient(1px 1px at 55% 55%, rgba(210,190,255,0.4) 0%, transparent 100%),
                    radial-gradient(1px 1px at 70% 20%, rgba(220,200,255,0.7) 0%, transparent 100%),
                    radial-gradient(1px 1px at 85% 70%, rgba(200,180,255,0.5) 0%, transparent 100%),
                    radial-gradient(1px 1px at 92% 35%, rgba(230,210,255,0.6) 0%, transparent 100%),
                    radial-gradient(1px 1px at 15% 75%, rgba(210,190,255,0.4) 0%, transparent 100%),
                    radial-gradient(1px 1px at 60% 90%, rgba(200,180,255,0.5) 0%, transparent 100%),
                    radial-gradient(1px 1px at 33% 60%, rgba(220,200,255,0.6) 0%, transparent 100%),
                    radial-gradient(2px 2px at 78% 5%,  rgba(200,160,255,0.8) 0%, transparent 100%),
                    radial-gradient(2px 2px at 48% 30%, rgba(180,140,255,0.7) 0%, transparent 100%),
                    radial-gradient(1px 1px at 5%  50%, rgba(220,200,255,0.5) 0%, transparent 100%),
                    radial-gradient(1px 1px at 95% 55%, rgba(210,190,255,0.6) 0%, transparent 100%),
                    radial-gradient(2px 2px at 22% 22%, rgba(200,160,255,0.7) 0%, transparent 100%);
                pointer-events: none;
                animation: twinkle 8s ease-in-out infinite alternate;
            }}

            @keyframes twinkle {{
                0%   {{ opacity: 0.6; }}
                100% {{ opacity: 1.0; }}
            }}

            /* ── Animated planes ── */
            .plane-layer {{
                position: fixed;
                inset: 0;
                z-index: 0;
                pointer-events: none;
                overflow: hidden;
            }}

            .plane {{
                position: absolute;
                opacity: 0.22;
                filter: drop-shadow(0 0 8px rgba(180,120,255,0.6));
                animation: fly linear infinite;
                white-space: nowrap;
                color: rgba(200,160,255,0.9);
            }}

            .plane:nth-child(1) {{ top:  8%; animation-duration: 28s; animation-delay:  0s;  font-size: 58px; opacity: 0.25; }}
            .plane:nth-child(2) {{ top: 25%; animation-duration: 40s; animation-delay: -14s; font-size: 42px; opacity: 0.18; }}
            .plane:nth-child(3) {{ top: 50%; animation-duration: 36s; animation-delay: -24s; font-size: 70px; opacity: 0.20; }}
            .plane:nth-child(4) {{ top: 68%; animation-duration: 32s; animation-delay:  -7s; font-size: 50px; opacity: 0.15; }}
            .plane:nth-child(5) {{ top: 84%; animation-duration: 48s; animation-delay: -38s; font-size: 36px; opacity: 0.13; }}

            @keyframes fly {{
                from {{ transform: translateX(-120px) translateY(0px); }}
                25%  {{ transform: translateX(25vw)   translateY(-8px); }}
                50%  {{ transform: translateX(50vw)   translateY(4px); }}
                75%  {{ transform: translateX(75vw)   translateY(-6px); }}
                to   {{ transform: translateX(110vw)  translateY(0px); }}
            }}

            /* ── Runway grid lines ── */
            .runway-grid {{
                position: fixed;
                bottom: 0; left: 0; right: 0;
                height: 120px;
                z-index: 0;
                pointer-events: none;
                background: repeating-linear-gradient(
                    90deg,
                    transparent,
                    transparent 48px,
                    rgba(160,80,255,0.06) 48px,
                    rgba(160,80,255,0.06) 50px
                );
                mask-image: linear-gradient(to top, rgba(0,0,0,0.3), transparent);
            }}

            /* ── All content above background ── */
            #react-entry-point {{ position: relative; z-index: 1; }}

            /* ── Scrollbar ── */
            ::-webkit-scrollbar {{ width: 5px; }}
            ::-webkit-scrollbar-track {{ background: {BG_MAIN}; }}
            ::-webkit-scrollbar-thumb {{ background: rgba(160,80,255,0.4); border-radius: 3px; }}

            /* ── Animations ── */
            @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.5; }} }}
            @keyframes fadeInUp {{
                from {{ opacity:0; transform:translateY(16px); }}
                to   {{ opacity:1; transform:translateY(0); }}
            }}
            @keyframes scanline {{
                0%   {{ transform: translateY(-100%); }}
                100% {{ transform: translateY(400%); }}
            }}

            .alert-pulse {{ animation: pulse 2s infinite; }}
            .fade-in      {{ animation: fadeInUp 0.5s ease both; }}

            /* ── Navbar glow ── */
            .aoip-navbar {{
                backdrop-filter: blur(12px);
                border-bottom: 1px solid rgba(160,80,255,0.2) !important;
                box-shadow: 0 1px 30px rgba(0,0,0,0.6), 0 0 60px rgba(0,80,160,0.1);
            }}

            /* ── Nav links ── */
            .nav-link {{
                position: relative;
                transition: color 0.2s;
                letter-spacing: 0.05em;
                font-size: 12px !important;
                text-transform: uppercase;
                font-family: 'Rajdhani', sans-serif;
                font-weight: 600;
            }}
            .nav-link:hover {{ color: #c090ff !important; }}
            .nav-link::after {{
                content: '';
                position: absolute;
                bottom: -2px; left: 0; right: 0;
                height: 1px;
                background: rgba(180,120,255,0.9);
                transform: scaleX(0);
                transition: transform 0.2s;
            }}
            .nav-link:hover::after {{ transform: scaleX(1); }}

            /* ── Cards ── */
            .aoip-card {{
                backdrop-filter: blur(8px);
                background: rgba(11,22,40,0.85) !important;
                border: 1px solid rgba(160,80,255,0.15) !important;
                transition: border-color 0.3s, box-shadow 0.3s;
            }}
            .aoip-card:hover {{
                border-color: rgba(160,80,255,0.35) !important;
                box-shadow: 0 8px 40px rgba(0,0,0,0.7), 0 0 24px rgba(160,80,255,0.10);
            }}

            /* ── KPI numbers ── */
            .kpi-value {{
                font-family: 'Share Tech Mono', monospace;
                letter-spacing: 0.02em;
            }}

            /* ── Section headings ── */
            h1, h2, h3 {{
                font-family: 'Rajdhani', sans-serif !important;
                letter-spacing: 0.06em;
            }}

            /* ── Buttons ── */
            button {{
                letter-spacing: 0.08em;
                text-transform: uppercase;
                font-family: 'Rajdhani', sans-serif !important;
                font-weight: 700 !important;
                transition: transform 0.15s, box-shadow 0.15s, filter 0.15s !important;
            }}
            button:hover {{
                transform: translateY(-1px);
                filter: brightness(1.15);
                box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
            }}
            button:active {{ transform: translateY(0px); }}

            /* ── Dash table ── */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {{
                background-color: rgba(15,30,53,0.9) !important;
                color: {TEXT_PRI} !important;
                border-color: rgba(160,80,255,0.10) !important;
                font-family: 'Inter', sans-serif !important;
                font-size: 12px !important;
            }}
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {{
                background-color: rgba(11,22,40,0.95) !important;
                color: {TEXT_HEAD} !important;
                border-color: rgba(160,80,255,0.18) !important;
                font-family: 'Rajdhani', sans-serif !important;
                font-size: 13px !important;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }}
/* ── Dropdown — always on top ── */
            .Select-control {{ border-radius: 4px !important; }}
           .Select-menu-outer {{
                z-index: 99999 !important;
                position: absolute !important;
                background-color: #1a0e30 !important;
                border: 1px solid rgba(160,80,255,0.3) !important;
                box-shadow: 0 8px 32px rgba(0,0,0,0.8) !important;
                bottom: 100% !important;
                top: auto !important;
                margin-bottom: 4px !important;
            }}
            .Select-option {{
                background-color: #1a0e30 !important;
                color: #d0c8e8 !important;
            }}
            .Select-option:hover,
            .Select-option.is-focused {{
                background-color: #2e1a50 !important;
                color: #c090ff !important;
            }}
            .Select-option.is-selected {{
                background-color: #3a1a60 !important;
                color: #c090ff !important;
            }}
            .Select.is-open {{ z-index: 99999 !important; }}
            .dash-dropdown {{ z-index: 9999 !important; position: relative !important; }}

            /* ── Scanline effect on cards (subtle) ── */
            .scanline-card {{ position: relative; overflow: hidden; }}
            .scanline-card::after {{
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0;
                height: 2px;
                background: linear-gradient(90deg, transparent, rgba(180,100,255,0.35), transparent);
                animation: scanline 4s linear infinite;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <!-- Animated plane layer -->
        <div class="plane-layer">
            <div class="plane">✈</div>
            <div class="plane">✈</div>
            <div class="plane">✈</div>
            <div class="plane">✈</div>
            <div class="plane">✈</div>
        </div>
        <!-- Runway grid at bottom -->
        <div class="runway-grid"></div>

        {{%app_entry%}}
        <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body>
</html>
'''

PAGE  = {'backgroundColor': 'transparent', 'color': TEXT_PRI, 'minHeight': '100vh', 'paddingBottom': '60px'}
CARD  = {'backgroundColor': 'rgba(11,22,40,0.82)', 'border': '1px solid rgba(0,200,160,0.12)',
         'borderRadius': '10px', 'padding': '28px', 'marginBottom': '24px',
         'boxShadow': '0 8px 32px rgba(0,0,0,0.6)', 'backdropFilter': 'blur(8px)',
         'className': 'aoip-card'}
PANEL = {'backgroundColor': 'rgba(15,30,53,0.75)', 'border': '1px solid rgba(0,200,160,0.10)',
         'borderRadius': '6px', 'padding': '20px', 'backdropFilter': 'blur(4px)'}


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
    link = {'color': TEXT_SEC, 'margin': '0 12px', 'textDecoration': 'none', 'fontSize': '12px', 'className': 'nav-link'}
    return html.Div(className='aoip-navbar', style={
        'backgroundColor': 'rgba(4,10,20,0.92)', 'padding': '14px 28px',
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'position': 'sticky', 'top': '0', 'zIndex': '100',
    }, children=[
        html.Span([
            html.Span("✈ ", style={'color': '#c090ff'}),
            html.Span("AOIP", style={'fontFamily': "'Rajdhani', sans-serif",
                                     'fontWeight': '700', 'fontSize': '18px',
                                     'letterSpacing': '0.12em', 'color': TEXT_HEAD}),
            html.Span(" · Airport Operations Intelligence",
                      style={'fontFamily': "'Inter', sans-serif",
                             'fontSize': '12px', 'color': TEXT_SEC,
                             'marginLeft': '8px', 'letterSpacing': '0.04em'}),
        ]),
        html.Div([
            dcc.Link("Home",           href="/",           style=link),
            dcc.Link("Passenger Flow", href="/passenger",  style=link),
            dcc.Link("Gate Opt.",      href="/gates",      style=link),
            dcc.Link("Risk",           href="/risk",       style=link),
            dcc.Link("Prediction",     href="/prediction", style=link),
            dcc.Link("Forecast",       href="/forecast",   style={**link, 'color': ACC_CYAN}),
            dcc.Link("Simulator",      href="/simulator",  style={**link, 'color': '#c47c2b'}),
            dcc.Link("History",        href="/history",    style={**link, 'color': ACC_TEAL}),
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
    P1 = "#b080ff"
    P2 = "#9060e8"
    P3 = "#c090ff"
    P4 = "#8050d0"
    P5 = "#d0a0ff"
    P6 = "#a070e8"
    P7 = "#7040c0"
    P8 = "#b890f0"
    nav_cards = [
        ("Passenger Flow",    P1, "/passenger",  "Real-time passenger movement and congestion heatmaps."),
        ("Gate Optimisation", P2, "/gates",      "Gate utilisation and AI-driven reassignment suggestions."),
        ("Risk Management",   P3, "/risk",       "Multi-factor operational risk scoring."),
        ("AI Predictions",    P4, "/prediction", "ML-powered delay forecasting with SHAP explainability."),
        ("Delay Forecast",    P5, "/forecast",   "Time-series delay forecast for next 24 hours and 7 days."),
        ("Prediction History",P6, "/history",    "All saved predictions with stats and charts."),
        ("Scenario Simulator",P7, "/simulator",  "Airport Digital Twin — simulate cascade delay effects."),
        ("Analytics",         P8, "/analytics",  "Weather impact and historical delay analytics."),
    ]
    return html.Div(style=PAGE, children=[
        navbar(),
        alert_banner(alerts),
        dcc.Interval(id='interval-refresh', interval=60*1000, n_intervals=0),
        html.Div(style={'padding': '40px', 'maxWidth': '1200px', 'margin': '0 auto'}, children=[
            html.H1("Airport Operations Intelligence Platform",
                    style={'color': TEXT_HEAD, 'fontWeight': '300', 'textAlign': 'center',
                           'marginBottom': '6px', 'fontFamily': "'Rajdhani', sans-serif",
                           'fontSize': '2.4rem', 'letterSpacing': '0.08em'}),
            html.P("Monitor · Analyse · Optimise — Real-time airport intelligence",
                   style={'color': '#4a90c4', 'textAlign': 'center', 'marginBottom': '10px',
                           'fontSize': '13px', 'letterSpacing': '0.15em', 'textTransform': 'uppercase'}),
            html.P(id='last-refresh',
                   style={'color': TEXT_SEC, 'textAlign': 'center', 'fontSize': '12px', 'marginBottom': '30px'}),
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginBottom': '40px'}, children=[
                kpi_card("Total Flights",     total_flights,     "#b080ff"),
                kpi_card("Avg Delay (min)",   avg_delay,         "#9060e8"),
                kpi_card("High Risk Flights", high_risk,         "#c090ff"),
                kpi_card("On-Time %",         f"{on_time_pct}%", "#a070e8"),
            ]),
            html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'},
                     children=[
                html.Div(style={
                    **CARD, 'width': '175px', 'textAlign': 'center',
                    'borderTop': f'3px solid {color}', 'marginBottom': '0'
                }, children=[
                    html.H3(label, style={'color': color, 'fontSize': '13px', 'marginTop': '0'}),
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
                    **PANEL, 'borderLeft': f'3px solid {pri_color.get(s["priority"], TEXT_SEC)}',
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
    airline_opts = (
        [{'label': a, 'value': a} for a in sorted(df_flights['airline'].unique())]
        if not df_flights.empty else [{'label': 'TunisAir', 'value': 'TunisAir'}]
    )
    RISK_PURPLE = '#9060e0'
    return html.Div(style=PAGE, children=[
        navbar(),
        html.Div(style={'padding': '30px 40px', 'maxWidth': '900px', 'margin': '0 auto'}, children=[
            html.H2("ML Operational Risk Engine", style={'color': ACC_PURP, 'fontWeight': '400'}),
            html.P("AI-powered risk scoring — Random Forest classifier with SHAP explainability.",
                   style={'color': TEXT_SEC}),
            html.Div(style=CARD, children=[
                html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'alignItems': 'flex-end'},
                         children=[
                    html.Div([
                        html.Label("Airline", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                     'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='risk-airline', options=airline_opts,
                                     value=airline_opts[0]['value'] if airline_opts else 'TunisAir',
                                     style={'width': '180px', 'color': '#000'})
                    ]),
                    html.Div([
                        html.Label("Terminal", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                      'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='risk-terminal', options=terminal_opts, value='T1',
                                     style={'width': '110px', 'color': '#000'})
                    ]),
                    html.Div([
                        html.Label("Weather", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                     'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='risk-weather',
                                     options=[{'label': w, 'value': w} for w in ['Clear','Cloudy','Rain','Storm']],
                                     value='Clear', style={'width': '130px', 'color': '#000'})
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
                        html.Label("Day", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                 'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='risk-day',
                                     options=[{'label': d, 'value': i} for i, d in enumerate(
                                         ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])],
                                     value=datetime.now().weekday(),
                                     style={'width': '100px', 'color': '#000'})
                    ]),
                    html.Button("Analyse Risk", id='calculate-risk-btn', n_clicks=0,
                                style={'backgroundColor': RISK_PURPLE, 'color': TEXT_HEAD, 'border': 'none',
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


def history_layout():
    """Prediction History page — pulls live data from the API."""
    return html.Div(style=PAGE, children=[
        navbar(),
        html.Div(style={'padding': '30px 40px', 'maxWidth': '1200px', 'margin': '0 auto'}, children=[
            html.H2("Prediction History", style={'color': ACC_TEAL, 'fontWeight': '400'}),
            html.P("All ML predictions saved to the database — with aggregate stats and trend charts.",
                   style={'color': TEXT_SEC}),

            html.Button("🔄 Refresh", id='history-refresh-btn', n_clicks=0,
                        style={'backgroundColor': ACC_TEAL, 'color': TEXT_HEAD, 'border': 'none',
                               'padding': '10px 22px', 'borderRadius': '5px',
                               'cursor': 'pointer', 'fontWeight': '600', 'marginBottom': '24px'}),

            # KPI stats row
            html.Div(id='history-stats'),

            # Charts row
            html.Div(id='history-charts'),

            # Table
            html.Div(style=CARD, children=[
                html.H3("Recent Predictions", style={'color': TEXT_HEAD, 'fontWeight': '400', 'marginTop': '0'}),
                html.Div(id='history-table')
            ])
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
# CALLBACKS — ROUTING
# =========================


ACC_ORANGE = '#c47c2b'

def simulator_layout():
    terminals = ['T1', 'T2', 'T3']
    weathers  = ['Clear', 'Cloudy', 'Rain', 'Storm']
    return html.Div(style=PAGE, children=[
        navbar(),
        html.Div(style={'padding': '30px 40px', 'maxWidth': '1200px', 'margin': '0 auto'}, children=[
            html.H2("✈ Airport Digital Twin — Scenario Simulator",
                    style={'color': ACC_ORANGE, 'fontWeight': '400'}),
            html.P("Simulate operational scenarios and predict cascade delay effects in real-time.",
                   style={'color': TEXT_SEC}),

            # ── Controls ─────────────────────────────────────
            html.Div(style=CARD, children=[
                html.H3("Scenario Parameters", style={'color': TEXT_HEAD, 'fontWeight': '400', 'marginTop': '0'}),
                html.Div(style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap', 'alignItems': 'flex-end'},
                         children=[
                    html.Div([
                        html.Label("Terminal", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                      'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='sim-terminal',
                                     options=[{'label': t, 'value': t} for t in terminals],
                                     value='T1', style={'width': '110px', 'color': '#000'})
                    ]),
                    html.Div([
                        html.Label("Weather", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                     'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(id='sim-weather',
                                     options=[{'label': w, 'value': w} for w in weathers],
                                     value='Clear', style={'width': '140px', 'color': '#000'})
                    ]),
                    html.Div([
                        html.Label("Number of Flights", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                               'display': 'block', 'marginBottom': '6px'}),
                        dcc.Slider(id='sim-flights', min=1, max=40, step=1, value=10,
                                   marks={1: '1', 10: '10', 20: '20', 30: '30', 40: '40'},
                                   tooltip={"placement": "bottom", "always_visible": True})
                    ], style={'flex': '1', 'minWidth': '220px'}),
                    html.Div([
                        html.Label("Start Hour", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                        'display': 'block', 'marginBottom': '6px'}),
                        dcc.Slider(id='sim-start', min=0, max=22, step=1, value=8,
                                   marks={0:'0', 6:'6', 12:'12', 18:'18', 22:'22'},
                                   tooltip={"placement": "bottom", "always_visible": True})
                    ], style={'flex': '1', 'minWidth': '200px'}),
                    html.Div([
                        html.Label("End Hour", style={'color': TEXT_SEC, 'fontSize': '12px',
                                                      'display': 'block', 'marginBottom': '6px'}),
                        dcc.Slider(id='sim-end', min=1, max=23, step=1, value=10,
                                   marks={1:'1', 6:'6', 12:'12', 18:'18', 23:'23'},
                                   tooltip={"placement": "bottom", "always_visible": True})
                    ], style={'flex': '1', 'minWidth': '200px'}),
                    html.Button("▶ Run Simulation", id='sim-run-btn', n_clicks=0,
                                style={'backgroundColor': ACC_ORANGE, 'color': TEXT_HEAD,
                                       'border': 'none', 'padding': '12px 28px',
                                       'borderRadius': '5px', 'cursor': 'pointer',
                                       'fontWeight': '700', 'fontSize': '14px'})
                ])
            ]),

            # ── Results ──────────────────────────────────────
            html.Div(id='sim-results')
        ])
    ])


@app.callback(
    Output('sim-results', 'children'),
    Input('sim-run-btn', 'n_clicks'),
    State('sim-terminal', 'value'),
    State('sim-weather',  'value'),
    State('sim-flights',  'value'),
    State('sim-start',    'value'),
    State('sim-end',      'value'),
)
def run_simulation(n_clicks, terminal, weather, flight_count, start_hour, end_hour):
    if not n_clicks:
        raise PreventUpdate

    if end_hour <= start_hour:
        return html.Div("⚠ End hour must be greater than start hour.",
                        style={'color': ACC_AMBER, 'padding': '16px'})

    try:
        response = httpx.post(f"{API_URL}/simulate", json={
            "terminal":     terminal,
            "weather":      weather,
            "flight_count": flight_count,
            "start_hour":   start_hour,
            "end_hour":     end_hour
        }, timeout=15)
        data = response.json()
    except Exception as ex:
        return html.Div(f"Simulation error: {ex}", style={'color': ACC_RED, 'padding': '20px'})

    summary  = data["summary"]
    metrics  = data["metrics"]
    hourly   = data["hourly_breakdown"]
    recs     = data["recommendations"]

    risk     = summary["risk_level"]
    risk_color = {'CRITICAL': '#8b1a1a', 'HIGH': ACC_RED,
                  'MEDIUM': ACC_AMBER, 'LOW': ACC_GREEN}.get(risk, ACC_BLUE)

    # ── Risk banner ──────────────────────────────────────
    banner = html.Div(style={
        'backgroundColor': risk_color, 'color': TEXT_HEAD,
        'padding': '14px 24px', 'borderRadius': '6px',
        'marginBottom': '24px', 'display': 'flex',
        'justifyContent': 'space-between', 'alignItems': 'center',
        **({"animation": "pulse 2s infinite"} if risk in ["CRITICAL","HIGH"] else {})
    }, children=[
        html.Div([
            html.Strong(f"SIMULATION RESULT — {risk} RISK  "),
            html.Span(f"{terminal} · {weather} · {flight_count} flights · "
                      f"{start_hour}:00–{end_hour}:00")
        ]),
        html.Div(f"Capacity used: {metrics['terminal_capacity_pct']}%",
                 style={'fontSize': '13px', 'opacity': '0.85'})
    ])

    # ── KPI row ──────────────────────────────────────────
    kpis = html.Div(style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap', 'marginBottom': '24px'},
                    children=[
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '130px',
                        'borderTop': f'3px solid {risk_color}', 'marginBottom': '0'}, children=[
            html.Div(f"{metrics['avg_delay_min']} min",
                     style={'fontSize': '28px', 'color': risk_color, 'fontWeight': '300'}),
            html.Div("Avg Delay", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '130px',
                        'borderTop': f'3px solid {ACC_AMBER}', 'marginBottom': '0'}, children=[
            html.Div(f"{metrics['cascade_delay_min']} min",
                     style={'fontSize': '28px', 'color': ACC_AMBER, 'fontWeight': '300'}),
            html.Div("Cascade Delay", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '130px',
                        'borderTop': f'3px solid {ACC_RED}', 'marginBottom': '0'}, children=[
            html.Div(f"{metrics['gate_overload_pct']}%",
                     style={'fontSize': '28px', 'color': ACC_RED, 'fontWeight': '300'}),
            html.Div("Gate Overload Prob.", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '130px',
                        'borderTop': f'3px solid {ACC_PURP}', 'marginBottom': '0'}, children=[
            html.Div(str(metrics['cancellations']),
                     style={'fontSize': '28px', 'color': ACC_PURP, 'fontWeight': '300'}),
            html.Div("Est. Cancellations", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '130px',
                        'borderTop': f'3px solid {ACC_BLUE}', 'marginBottom': '0'}, children=[
            html.Div(f"{metrics['affected_passengers']:,}",
                     style={'fontSize': '28px', 'color': ACC_BLUE, 'fontWeight': '300'}),
            html.Div("Affected Passengers", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '130px',
                        'borderTop': f'3px solid {ACC_TEAL}', 'marginBottom': '0'}, children=[
            html.Div(f"{metrics['terminal_capacity_pct']}%",
                     style={'fontSize': '28px', 'color': ACC_TEAL, 'fontWeight': '300'}),
            html.Div("Capacity Used", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
    ])

    # ── Hourly chart ─────────────────────────────────────
    hours      = [f"{h['hour']}:00" for h in hourly]
    delays     = [h['avg_delay']     for h in hourly]
    gate_loads = [h['gate_load_pct'] for h in hourly]
    bar_colors = [ACC_RED if h['status'] == 'OVERLOADED'
                  else ACC_AMBER if h['status'] == 'HIGH'
                  else ACC_GREEN for h in hourly]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hours, y=delays, name='Avg Delay (min)',
        marker_color=bar_colors,
        text=[f"{d:.0f}m" for d in delays], textposition='outside'
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=gate_loads, name='Gate Load %',
        mode='lines+markers', yaxis='y2',
        line=dict(color=ACC_CYAN, width=2),
        marker=dict(size=7)
    ))
    fig.update_layout(
        title=f'Hourly Breakdown — {terminal} · {weather} · {flight_count} flights',
        xaxis_title='Hour', yaxis_title='Avg Delay (min)',
        yaxis2=dict(title='Gate Load %', overlaying='y', side='right', range=[0, 120]),
        height=380, hovermode='x unified', barmode='group',
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI,
        legend=dict(bgcolor=BG_PANEL, bordercolor=BORDER)
    )

    # ── Recommendations ──────────────────────────────────
    pri_color = {'CRITICAL': '#8b1a1a', 'HIGH': ACC_RED, 'MEDIUM': ACC_AMBER, 'LOW': ACC_GREEN}
    rec_cards = [
        html.Div(style={
            **PANEL,
            'borderLeft': f'3px solid {pri_color.get(r["priority"], ACC_BLUE)}',
            'marginBottom': '10px',
            'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'
        }, children=[
            html.Div([
                html.Div(r['action'], style={'color': TEXT_HEAD, 'fontWeight': '500'}),
                html.Div(r['impact'], style={'color': TEXT_SEC, 'fontSize': '12px', 'marginTop': '4px'})
            ]),
            html.Span(r['priority'],
                      style={'color': pri_color.get(r['priority'], ACC_BLUE),
                             'fontWeight': '700', 'fontSize': '12px', 'minWidth': '70px',
                             'textAlign': 'right'})
        ]) for r in recs
    ]

    return html.Div([
        banner,
        kpis,
        html.Div(style=CARD, children=[dcc.Graph(figure=fig)]),
        html.Div(style=CARD, children=[
            html.H3("🤖 AI Recommendations", style={'color': ACC_ORANGE, 'fontWeight': '400', 'marginTop': '0'}),
            html.Div(rec_cards)
        ]),
        html.Div(
            f"Simulated at {data['simulated_at'][:19].replace('T',' ')} · "
            f"Gate capacity: {summary['gate_count']} gates · "
            f"Max capacity: {summary['capacity']} flights",
            style={'color': TEXT_SEC, 'fontSize': '11px', 'textAlign': 'right', 'marginTop': '8px'}
        )
    ])


@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/passenger': return passenger_flow_layout()
    if pathname == '/gates':     return gate_optimization_layout()
    if pathname == '/risk':      return risk_management_layout()
    if pathname == '/prediction':return prediction_layout()
    if pathname == '/forecast':  return forecast_layout()
    if pathname == '/simulator': return simulator_layout()
    if pathname == '/history':   return history_layout()
    if pathname == '/analytics': return analytics_layout()
    return home_layout()


@app.callback(Output('last-refresh', 'children'), Input('interval-refresh', 'n_intervals'))
def update_refresh_time(n):
    return f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}  ·  Auto-refresh every 60s"


@app.callback(Output('passenger-heatmap', 'figure'), Input('terminal-select', 'value'))
def update_heatmap(terminal):
    return passenger_ai.get_heatmap(terminal)


# =========================
# HISTORY CALLBACKS
# =========================
@app.callback(
    Output('history-stats',  'children'),
    Output('history-charts', 'children'),
    Output('history-table',  'children'),
    Input('history-refresh-btn', 'n_clicks')
)
def load_history(n_clicks):
    try:
        stats_resp   = httpx.get(f"{API_URL}/history/predictions/stats",   timeout=10).json()
        airline_resp = httpx.get(f"{API_URL}/history/predictions/by-airline", timeout=10).json()
        hour_resp    = httpx.get(f"{API_URL}/history/predictions/by-hour",  timeout=10).json()
        preds_resp   = httpx.get(f"{API_URL}/history/predictions?limit=100", timeout=10).json()
    except Exception as ex:
        msg = html.Div(f"Could not load history: {ex} — Run some predictions first!",
                       style={'color': ACC_AMBER, 'padding': '20px'})
        return msg, html.Div(), html.Div()

    total        = stats_resp.get("total", 0)
    avg_delay    = stats_resp.get("avg_delay", 0) or 0
    high_count   = stats_resp.get("high_risk_count", 0) or 0
    low_count    = stats_resp.get("low_risk_count",  0) or 0

    # ── KPI row ───────────────────────────────────────────────
    stats_row = html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginBottom': '24px'},
                         children=[
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '140px',
                        'borderTop': f'3px solid {ACC_BLUE}', 'marginBottom': '0'}, children=[
            html.Div(str(total), style={'fontSize': '32px', 'color': ACC_BLUE, 'fontWeight': '300'}),
            html.Div("Total Predictions", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '140px',
                        'borderTop': f'3px solid {ACC_AMBER}', 'marginBottom': '0'}, children=[
            html.Div(f"{avg_delay} min", style={'fontSize': '32px', 'color': ACC_AMBER, 'fontWeight': '300'}),
            html.Div("Avg Predicted Delay", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '140px',
                        'borderTop': f'3px solid {ACC_RED}', 'marginBottom': '0'}, children=[
            html.Div(str(high_count), style={'fontSize': '32px', 'color': ACC_RED, 'fontWeight': '300'}),
            html.Div("High Risk Predictions", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
        html.Div(style={**CARD, 'textAlign': 'center', 'flex': '1', 'minWidth': '140px',
                        'borderTop': f'3px solid {ACC_GREEN}', 'marginBottom': '0'}, children=[
            html.Div(str(low_count), style={'fontSize': '32px', 'color': ACC_GREEN, 'fontWeight': '300'}),
            html.Div("Low Risk Predictions", style={'color': TEXT_SEC, 'fontSize': '12px'})
        ]),
    ])

    # ── Charts ────────────────────────────────────────────────
    charts = html.Div()
    airline_data = airline_resp.get("data", [])
    hour_data    = hour_resp.get("data", [])

    if airline_data:
        df_a = pd.DataFrame(airline_data)
        fig_a = px.bar(df_a, x='airline', y='avg_delay', color='avg_delay',
                       title='Avg Predicted Delay by Airline (from DB)',
                       color_continuous_scale='Reds')
        fig_a.update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)

        fig_count = px.bar(df_a, x='airline', y='total', color='total',
                           title='Prediction Count by Airline',
                           color_continuous_scale='Blues')
        fig_count.update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)

        charts_row1 = html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
            html.Div(style={**CARD, 'flex': '1', 'minWidth': '400px'}, children=[dcc.Graph(figure=fig_a)]),
            html.Div(style={**CARD, 'flex': '1', 'minWidth': '400px'}, children=[dcc.Graph(figure=fig_count)]),
        ])

        if hour_data:
            df_h = pd.DataFrame(hour_data)
            fig_h = px.line(df_h, x='hour', y='avg_delay',
                            title='Avg Predicted Delay by Hour (from DB)',
                            markers=True, color_discrete_sequence=[ACC_CYAN])
            fig_h.update_layout(paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI)
            charts_row2 = html.Div(style=CARD, children=[dcc.Graph(figure=fig_h)])
        else:
            charts_row2 = html.Div()

        charts = html.Div([charts_row1, charts_row2])

    # ── Table ─────────────────────────────────────────────────
    predictions = preds_resp.get("predictions", [])
    if not predictions:
        table = html.Div("No predictions yet — go to the Prediction page and run some!",
                         style={'color': TEXT_SEC, 'fontSize': '14px'})
    else:
        table_data = []
        for p in predictions[:50]:
            table_data.append({
                "Time":        p.get("created_at", "")[:19].replace("T", " "),
                "Airline":     p.get("airline", ""),
                "Terminal":    p.get("terminal", ""),
                "Weather":     p.get("weather", ""),
                "Day":         p.get("day", ""),
                "Delay (min)": p.get("predicted_delay", ""),
                "Probability": f"{p.get('delay_probability', '')}%",
                "Risk":        p.get("risk_level", ""),
                "Model":       p.get("model_used", "")
            })

        table = dash_table.DataTable(
            data=table_data,
            columns=[{"name": c, "id": c} for c in table_data[0].keys()],
            style_table={'overflowX': 'auto'},
            style_cell={'backgroundColor': BG_PANEL, 'color': TEXT_PRI,
                        'border': f'1px solid {BORDER}', 'padding': '10px',
                        'fontSize': '13px', 'textAlign': 'left'},
            style_header={'backgroundColor': BG_CARD, 'color': TEXT_HEAD,
                          'fontWeight': '600', 'border': f'1px solid {BORDER}'},
            style_data_conditional=[
                {'if': {'filter_query': '{Risk} = "HIGH"'},
                 'color': ACC_RED, 'fontWeight': '600'},
                {'if': {'filter_query': '{Risk} = "MEDIUM"'},
                 'color': ACC_AMBER},
                {'if': {'filter_query': '{Risk} = "LOW"'},
                 'color': ACC_GREEN},
            ],
            page_size=20,
            sort_action='native',
            filter_action='native',
        )

    return stats_row, charts, table


# =========================
# FORECAST CALLBACK
# =========================
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
            "weather": weather, "hours_ahead": hours_ahead
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
        title=f'Delay Forecast — Next {hours_ahead} Hours ({weather})',
        xaxis_title='Time', yaxis_title='Average Delay (min)',
        height=420, hovermode='x unified',
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI,
        legend=dict(bgcolor=BG_PANEL, bordercolor=BORDER)
    )

    bar_colors = [ACC_RED if v > 40 else ACC_AMBER if v > 25 else ACC_GREEN for v in weekly["values"]]
    fig_weekly = go.Figure(go.Bar(
        x=weekly["days"], y=weekly["values"], marker_color=bar_colors,
        text=[f"{v:.0f} min" for v in weekly["values"]], textposition='outside'
    ))
    fig_weekly.update_layout(
        title='7-Day Delay Outlook', xaxis_title='Day', yaxis_title='Expected Avg Delay (min)',
        height=320, paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI
    )

    warning_cards = [
        html.Div(style={
            **PANEL, 'borderLeft': f'3px solid {ACC_RED if w["level"] == "CRITICAL" else ACC_AMBER}',
            'marginBottom': '8px', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'
        }, children=[
            html.Span(f"⚠ {w['message']}", style={'color': TEXT_HEAD, 'fontSize': '13px'}),
            html.Span(w['level'], style={'color': ACC_RED if w['level'] == 'CRITICAL' else ACC_AMBER,
                                         'fontWeight': '600', 'fontSize': '12px'})
        ]) for w in warnings[:4]
    ]

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


# =========================
# PREDICTION CALLBACK
# =========================
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


# =========================
# RISK CALLBACK
# =========================
@app.callback(
    Output('risk-calculation-result', 'children'),
    Input('calculate-risk-btn', 'n_clicks'),
    State('risk-airline',  'value'),
    State('risk-terminal', 'value'),
    State('risk-weather',  'value'),
    State('risk-hour',     'value'),
    State('risk-day',      'value'),
)
def calculate_risk(n_clicks, airline, terminal, weather, hour, day_of_week):
    if not n_clicks:
        raise PreventUpdate

    # Call ML risk API
    try:
        resp = httpx.post(f"{API_URL}/risk", json={
            "terminal":    terminal,
            "hour":        int(hour),
            "weather":     weather,
            "airline":     airline,
            "day_of_week": int(day_of_week)
        }, timeout=10)
        data       = resp.json()
        score      = data["risk_score"]
        level      = data["risk_level"]
        model_used = data.get("model_used", "unknown")
        shap_data  = data.get("shap_explanation", [])
        factors    = data.get("factors", {})
    except Exception as ex:
        # Local fallback
        score      = risk_ai.calculate_risk_score(terminal, hour, weather)
        level      = "LOW" if score < 40 else "MEDIUM" if score < 70 else "HIGH"
        model_used = f"local fallback ({ex})"
        shap_data  = []
        factors    = {}

    color_map = {"LOW": "#7040c0", "MEDIUM": "#9060e0", "HIGH": "#c090ff"}
    color = color_map.get(level, ACC_PURP)
    icon  = {"LOW": "●", "MEDIUM": "◉", "HIGH": "⬤"}.get(level, "●")

    # SHAP bar chart
    shap_section = html.Div()
    if shap_data:
        feat_names = [s['feature']             for s in shap_data]
        impacts    = [s['impact']               for s in shap_data]
        directions = [s['direction']            for s in shap_data]
        bar_colors = ['#e050a0' if v > 0 else '#7040c0' for v in impacts]

        fig_shap = go.Figure(go.Bar(
            x=impacts, y=feat_names, orientation='h',
            marker_color=bar_colors,
            text=[f"{'▲' if v>0 else '▼'} {abs(v):.3f}" for v in impacts],
            textposition='outside'
        ))
        fig_shap.update_layout(
            title='SHAP Feature Contributions — Why this risk level?',
            xaxis_title='Impact on Risk Score',
            height=320,
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_PANEL, font_color=TEXT_PRI,
            margin=dict(l=120, r=60, t=50, b=40)
        )
        shap_section = html.Div(style=CARD, children=[
            html.H3("🔍 ML Explainability — SHAP Analysis",
                    style={'color': ACC_PURP, 'fontWeight': '400', 'marginTop': '0'}),
            html.P(f"Model: {model_used}",
                   style={'color': TEXT_SEC, 'fontSize': '12px', 'marginBottom': '12px'}),
            dcc.Graph(figure=fig_shap),
            html.Div([
                html.Div(style={
                    **PANEL, 'borderLeft': f'3px solid {"#e050a0" if s["impact"]>0 else "#7040c0"}',
                    'marginBottom': '8px', 'display': 'flex',
                    'justifyContent': 'space-between'
                }, children=[
                    html.Span(s['feature'], style={'color': TEXT_HEAD, 'fontSize': '13px'}),
                    html.Span(f"{'▲' if s['impact']>0 else '▼'} {abs(s['impact']):.3f}  {s['direction']}",
                              style={'color': '#e050a0' if s['impact']>0 else '#7040c0',
                                     'fontSize': '12px'})
                ]) for s in shap_data[:6]
            ])
        ])

    high_alert = html.Div()
    if level == "HIGH":
        high_alert = html.Div(className='alert-pulse', style={
            'background': 'linear-gradient(135deg, #3a0060, #200040)',
            'border': '1px solid #c090ff',
            'color': TEXT_HEAD, 'padding': '14px 20px',
            'borderRadius': '8px', 'marginBottom': '16px', 'fontWeight': '600',
            'boxShadow': '0 0 20px rgba(192,144,255,0.3)'
        }, children=f"🚨 HIGH RISK — {airline} · {terminal} · {weather} — ML model confidence HIGH")

    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    day_label = days[int(day_of_week)] if day_of_week is not None else '—'

    return html.Div([
        high_alert,
        html.Div(style=CARD, children=[
            html.H3("Risk Assessment Result",
                    style={'color': TEXT_HEAD, 'fontWeight': '400', 'marginTop': '0',
                           'marginBottom': '20px'}),
            html.Div(style={'display': 'flex', 'gap': '24px', 'alignItems': 'center', 'flexWrap': 'wrap'},
                     children=[
                # Score circle
                html.Div(style={
                    'textAlign': 'center', 'minWidth': '150px',
                    'padding': '20px',
                    'borderRadius': '50%',
                    'background': f'radial-gradient(circle, rgba(144,96,224,0.15), rgba(8,5,15,0))',
                    'border': f'2px solid {color}',
                    'boxShadow': f'0 0 30px {color}40'
                }, children=[
                    html.Div(icon, style={'fontSize': '24px', 'color': color}),
                    html.Div(f"{score:.0f}",
                             style={'fontSize': '52px', 'color': color,
                                    'fontWeight': '200', 'lineHeight': '1',
                                    'fontFamily': "'Share Tech Mono', monospace"}),
                    html.Div("/100", style={'color': TEXT_SEC, 'fontSize': '12px'}),
                    html.Div(level, style={'color': color, 'fontWeight': '700',
                                          'fontSize': '14px', 'marginTop': '6px',
                                          'letterSpacing': '0.1em'})
                ]),
                # Details panel
                html.Div(style={**PANEL, 'flex': '1', 'minWidth': '200px'}, children=[
                    html.Div("Input Parameters",
                             style={'color': TEXT_SEC, 'fontSize': '11px',
                                    'letterSpacing': '0.1em', 'textTransform': 'uppercase',
                                    'marginBottom': '12px'}),
                    html.Table([
                        html.Tr([
                            html.Td(k, style={'color': TEXT_SEC, 'paddingRight': '16px',
                                              'paddingBottom': '8px', 'fontSize': '12px'}),
                            html.Td(v, style={'color': TEXT_HEAD, 'fontWeight': '500', 'fontSize': '13px'})
                        ]) for k, v in [
                            ("Airline",  airline),
                            ("Terminal", terminal),
                            ("Weather",  weather),
                            ("Hour",     f"{hour}:00"),
                            ("Day",      day_label),
                            ("Model",    model_used[:30] + "..." if len(str(model_used)) > 30 else str(model_used))
                        ]
                    ])
                ]),
                # Actions panel
                html.Div(style={**PANEL, 'flex': '2', 'minWidth': '200px'}, children=[
                    html.Div("Recommended Actions",
                             style={'color': TEXT_SEC, 'fontSize': '11px',
                                    'letterSpacing': '0.1em', 'textTransform': 'uppercase',
                                    'marginBottom': '12px'}),
                    html.Ul([
                        html.Li("Increase monitoring frequency",
                                style={'color': TEXT_PRI, 'marginBottom': '6px', 'fontSize': '13px'}),
                        html.Li("Prepare backup staff if score > 60",
                                style={'color': TEXT_PRI, 'marginBottom': '6px', 'fontSize': '13px'}),
                        html.Li("Notify airline operations if score > 75",
                                style={'color': TEXT_PRI, 'marginBottom': '6px', 'fontSize': '13px'}),
                        html.Li("Activate contingency plan if score > 85",
                                style={'color': TEXT_PRI, 'fontSize': '13px'}),
                    ], style={'margin': '0', 'paddingLeft': '18px'})
                ])
            ])
        ]),
        shap_section
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