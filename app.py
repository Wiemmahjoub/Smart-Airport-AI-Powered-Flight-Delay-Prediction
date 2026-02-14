import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
from datetime import datetime, timedelta

# =========================
# LOAD DATA & MODEL
# =========================
# Add these imports at the TOP of your app.py
from passenger_flow import PassengerFlowAnalyzer
from gate_optimizer import GateOptimizer

# Initialize analyzers
passenger_analyzer = PassengerFlowAnalyzer()
gate_optimizer = GateOptimizer()
try:
    df_flights = pd.read_csv("data/processed/flights_clean.csv")
    df_flights.columns = df_flights.columns.str.lower().str.strip()
    
    if 'departure_time' in df_flights.columns:
        df_flights['departure_time'] = pd.to_datetime(df_flights['departure_time'])
        df_flights['departure_hour'] = df_flights['departure_time'].dt.hour
        df_flights['departure_day'] = df_flights['departure_time'].dt.date
        df_flights['departure_week'] = df_flights['departure_time'].dt.isocalendar().week
    
    if 'delay_minutes' in df_flights.columns:
        def categorize_risk(delay):
            if delay <= 15: return "Low"
            elif delay <= 30: return "Medium"
            else: return "High"
        df_flights['delay_risk'] = df_flights['delay_minutes'].apply(categorize_risk)
    
except Exception as e:
    print(f"Error loading flights data: {e}")
    df_flights = pd.DataFrame(columns=['airline', 'terminal', 'day_of_week', 'weather', 'delay_minutes'])

try:
    model = joblib.load("model/delay_model.pkl")
except:
    print("Warning: Model not found. Using dummy model.")
    model = None

# =========================
# DASH APP
# =========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Smart Airport AI Dashboard"

# =========================
# ADVANCED 3D COLOR SCHEME
# =========================
COLORS = {
    'primary_dark': '#0f172a',
    'primary': '#1e293b',
    'primary_light': '#334155',
    'secondary_dark': '#0369a1',
    'secondary': '#0ea5e9',
    'secondary_light': '#38bdf8',
    'accent_dark': '#0d9488',
    'accent': '#14b8a6',
    'accent_light': '#2dd4bf',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'gradient_start': '#0f172a',
    'gradient_mid': '#1e293b',
    'gradient_end': '#334155',
    'chart_3d_1': '#0ea5e9',
    'chart_3d_2': '#8b5cf6',
    'chart_3d_3': '#10b981',
    'chart_3d_4': '#f59e0b',
    'glass_dark': 'rgba(30, 41, 59, 0.8)',
    'glass_light': 'rgba(255, 255, 255, 0.1)',
}

# =========================
# ADVANCED 3D ANIMATED STYLES
# =========================
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            :root {
                --primary-dark: #0f172a;
                --primary: #1e293b;
                --primary-light: #334155;
                --secondary: #0ea5e9;
                --accent: #14b8a6;
                --glass: rgba(255, 255, 255, 0.05);
                --glow: 0 0 20px rgba(14, 165, 233, 0.3);
            }
            
            body {
                background: linear-gradient(135deg, var(--primary-dark), var(--primary));
                color: #e2e8f0;
                min-height: 100vh;
                overflow-x: hidden;
                position: relative;
            }
            
            #particles-js {
                position: fixed;
                width: 100%;
                height: 100%;
                z-index: -1;
            }
            
            .glass-card {
                background: linear-gradient(135deg, 
                    rgba(30, 41, 59, 0.7), 
                    rgba(30, 41, 59, 0.4));
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            
            .glass-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(
                    90deg,
                    transparent,
                    rgba(255, 255, 255, 0.1),
                    transparent
                );
                transition: 0.5s;
            }
            
            .glass-card:hover::before {
                left: 100%;
            }
            
            .glass-card:hover {
                transform: translateY(-5px) scale(1.02);
                box-shadow: 
                    0 12px 40px rgba(0, 0, 0, 0.4),
                    0 0 30px rgba(14, 165, 233, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                border-color: rgba(14, 165, 233, 0.3);
            }
            
            .kpi-card-3d {
                background: linear-gradient(145deg, #1e293b, #0f172a);
                border-radius: 16px;
                padding: 24px;
                position: relative;
                transform-style: preserve-3d;
                perspective: 1000px;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 
                    -8px -8px 16px rgba(255, 255, 255, 0.05),
                    8px 8px 16px rgba(0, 0, 0, 0.5),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
            
            .kpi-card-3d:hover {
                transform: translateY(-8px) rotateX(5deg);
                box-shadow: 
                    -12px -12px 24px rgba(255, 255, 255, 0.08),
                    12px 12px 24px rgba(0, 0, 0, 0.6),
                    0 0 30px rgba(14, 165, 233, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }
            
            .kpi-value-3d {
                font-size: 36px;
                font-weight: 800;
                background: linear-gradient(135deg, #0ea5e9, #38bdf8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-shadow: 0 2px 10px rgba(14, 165, 233, 0.3);
                margin: 12px 0;
                position: relative;
                z-index: 2;
            }
            
            .nav-3d {
                background: linear-gradient(90deg, 
                    rgba(15, 23, 42, 0.9), 
                    rgba(30, 41, 59, 0.9));
                backdrop-filter: blur(20px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
                position: relative;
                z-index: 1000;
            }
            
            .nav-link-3d {
                color: #94a3b8;
                text-decoration: none;
                font-weight: 600;
                font-size: 14px;
                padding: 12px 20px;
                margin: 0 4px;
                border-radius: 12px;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .nav-link-3d::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(14, 165, 233, 0.2), transparent);
                transition: 0.5s;
            }
            
            .nav-link-3d:hover::before {
                left: 100%;
            }
            
            .nav-link-3d:hover {
                color: #0ea5e9;
                background: rgba(14, 165, 233, 0.1);
                transform: translateY(-2px);
            }
            
            .nav-link-3d.active {
                color: #0ea5e9;
                background: rgba(14, 165, 233, 0.15);
                box-shadow: 0 4px 15px rgba(14, 165, 233, 0.2);
            }
            
            .btn-3d {
                background: linear-gradient(135deg, #0ea5e9, #0369a1);
                color: white;
                border: none;
                padding: 14px 32px;
                border-radius: 12px;
                font-weight: 600;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
                display: inline-flex;
                align-items: center;
                gap: 12px;
                box-shadow: 
                    0 6px 20px rgba(14, 165, 233, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
                transform: translateY(0);
            }
            
            .btn-3d::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: 0.5s;
            }
            
            .btn-3d:hover::before {
                left: 100%;
            }
            
            .btn-3d:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 
                    0 10px 30px rgba(14, 165, 233, 0.6),
                    0 0 40px rgba(14, 165, 233, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
            }
            
            .input-3d {
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 14px 16px;
                color: #e2e8f0;
                font-size: 14px;
                transition: all 0.3s ease;
                width: 100%;
                backdrop-filter: blur(10px);
            }
            
            .input-3d:focus {
                outline: none;
                border-color: #0ea5e9;
                box-shadow: 
                    0 0 0 3px rgba(14, 165, 233, 0.1),
                    0 4px 20px rgba(14, 165, 233, 0.2);
                transform: translateY(-2px);
            }
            
            .grid-lines {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: 
                    linear-gradient(rgba(14, 165, 233, 0.1) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(14, 165, 233, 0.1) 1px, transparent 1px);
                background-size: 50px 50px;
                pointer-events: none;
                z-index: -1;
                opacity: 0.3;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                50% { transform: translateY(-10px) rotate(1deg); }
            }
            
            @keyframes pulse-glow {
                0%, 100% { box-shadow: 0 0 20px rgba(14, 165, 233, 0.3); }
                50% { box-shadow: 0 0 40px rgba(14, 165, 233, 0.6); }
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .loading-spinner {
                border: 3px solid rgba(255, 255, 255, 0.1);
                border-top: 3px solid #0ea5e9;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }
            
            .gauge-3d {
                position: relative;
                animation: float 6s ease-in-out infinite;
            }
            
            .pulse {
                animation: pulse-glow 2s infinite;
            }
            
            .container-3d {
                max-width: 1400px;
                margin: 0 auto;
                padding: 30px 20px;
            }
            
            @media (max-width: 768px) {
                .kpi-value-3d {
                    font-size: 28px;
                }
                .btn-3d {
                    padding: 12px 24px;
                }
                .container-3d {
                    padding: 20px 15px;
                }
            }
            
            ::-webkit-scrollbar {
                width: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(15, 23, 42, 0.5);
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, #0ea5e9, #0369a1);
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <div id="particles-js"></div>
        <div class="grid-lines"></div>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
        <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
        <script>
            particlesJS('particles-js', {
                particles: {
                    number: { value: 80, density: { enable: true, value_area: 800 } },
                    color: { value: "#0ea5e9" },
                    shape: { type: "circle" },
                    opacity: { value: 0.3, random: true },
                    size: { value: 3, random: true },
                    line_linked: {
                        enable: true,
                        distance: 150,
                        color: "#0ea5e9",
                        opacity: 0.1,
                        width: 1
                    },
                    move: {
                        enable: true,
                        speed: 2,
                        direction: "none",
                        random: true,
                        straight: false,
                        out_mode: "out",
                        bounce: false
                    }
                },
                interactivity: {
                    detect_on: "canvas",
                    events: {
                        onhover: { enable: true, mode: "repulse" },
                        onclick: { enable: true, mode: "push" },
                        resize: true
                    }
                },
                retina_detect: true
            });
        </script>
    </body>
</html>
'''

# =========================
# 3D NAVIGATION BAR
# =========================
def get_3d_navbar():
    return html.Div(className="nav-3d", style={
        "padding": "0 40px",
        "height": "70px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "position": "sticky",
        "top": "0",
        "zIndex": "1000"
    }, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "30px"}, children=[
            html.Div(style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "fontSize": "20px",
                "fontWeight": "700",
                "background": "linear-gradient(135deg, #0ea5e9, #14b8a6)",
                "WebkitBackgroundClip": "text",
                "WebkitTextFillColor": "transparent",
                "backgroundClip": "text"
            }, children=[
                html.I(className="fas fa-plane-departure", style={"fontSize": "24px"}),
                "AERO AI DASHBOARD"
            ]),
            
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                dcc.Link(html.Div([
                    html.I(className="fas fa-chart-line", style={"marginRight": "8px"}),
                    "Dashboard"
                ]), href="/", className="nav-link-3d", id="nav-dashboard"),
                
                dcc.Link(html.Div([
                    html.I(className="fas fa-brain", style={"marginRight": "8px"}),
                    "AI Predictions"
                ]), href="/prediction", className="nav-link-3d", id="nav-prediction"),
                
                dcc.Link(html.Div([
                    html.I(className="fas fa-star", style={"marginRight": "8px"}),
                    "Best Flights"
                ]), href="/best-flight", className="nav-link-3d", id="nav-best-flight"),
                
                dcc.Link(html.Div([
                    html.I(className="fas fa-chart-bar", style={"marginRight": "8px"}),
                    "Analytics"
                ]), href="/analytics", className="nav-link-3d", id="nav-analytics"),
            ])
        ]),
        
        html.Div(style={
            "display": "flex",
            "alignItems": "center",
            "gap": "20px"
        }, children=[
            html.Div(style={
                "padding": "8px 16px",
                "background": "rgba(14, 165, 233, 0.1)",
                "borderRadius": "12px",
                "border": "1px solid rgba(14, 165, 233, 0.2)",
                "fontSize": "12px",
                "fontWeight": "600",
                "color": "#0ea5e9",
                "display": "flex",
                "alignItems": "center",
                "gap": "8px"
            }, children=[
                html.I(className="fas fa-wifi"),
                "LIVE DATA STREAMING"
            ]),
            
            html.Div(style={
                "width": "12px",
                "height": "12px",
                "background": "#10b981",
                "borderRadius": "50%",
                "animation": "pulse-glow 2s infinite"
            })
        ])
    ])

# =========================
# CREATE 3D CHARTS (FIXED VERSION)
# =========================
def create_3d_scatter():
    """Create a 3D scatter plot for flight patterns"""
    np.random.seed(42)
    n_points = 50
    
    # Generate sample 3D data
    x = np.random.normal(0, 1, n_points)
    y = np.random.normal(0, 1, n_points)
    z = np.random.normal(0, 1, n_points)
    colors = np.random.choice(['#0ea5e9', '#8b5cf6', '#10b981'], n_points)
    sizes = np.random.uniform(5, 15, n_points)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
            line=dict(width=0),
            symbol='circle'
        ),
        hovertemplate='<b>Flight Pattern</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                backgroundcolor='rgba(0,0,0,0)',
                zerolinecolor='rgba(255,255,255,0.3)'
            ),
            yaxis=dict(
                title='',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                backgroundcolor='rgba(0,0,0,0)',
                zerolinecolor='rgba(255,255,255,0.3)'
            ),
            zaxis=dict(
                title='',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                backgroundcolor='rgba(0,0,0,0)',
                zerolinecolor='rgba(255,255,255,0.3)'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        height=300,
        showlegend=False
    )
    
    return fig

def create_3d_surface():
    """Create a 3D surface plot for delay patterns"""
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, '#0f172a'], [0.5, '#0ea5e9'], [1, '#14b8a6']],
        opacity=0.9,
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="#14b8a6",
                project=dict(z=True)
            )
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            zaxis=dict(showgrid=False, visible=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        height=300,
        showlegend=False
    )
    
    return fig

def create_3d_bar():
    """Create a 3D bar chart for airline performance - FIXED VERSION"""
    airlines = df_flights['airline'].unique()[:5] if not df_flights.empty else ['Air France', 'Emirates', 'Lufthansa', 'Qatar', 'TunisAir']
    terminals = ['T1', 'T2', 'T3']
    
    # Generate sample data
    np.random.seed(42)
    data = []
    for i, airline in enumerate(airlines):
        for j, terminal in enumerate(terminals):
            delay = np.random.uniform(10, 40)
            data.append(dict(
                airline=airline,
                terminal=terminal,
                delay=delay,
                x=i,
                y=j,
                z=delay
            ))
    
    df_3d = pd.DataFrame(data)
    
    # Create 3D bar chart using regular 3D scatter with custom markers
    fig = go.Figure()
    
    # Add each bar as a separate trace for better control
    for idx, row in df_3d.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['x'], row['x']],
            y=[row['y'], row['y']],
            z=[0, row['z']],
            mode='lines',
            line=dict(
                color=row['delay'],
                colorscale=[[0, '#0f172a'], [0.5, '#0ea5e9'], [1, '#14b8a6']],
                width=8
            ),
            showlegend=False,
            hoverinfo='text',
            text=f"Airline: {row['airline']}<br>Terminal: {row['terminal']}<br>Delay: {row['delay']:.1f} min"
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Airline',
                ticktext=airlines,
                tickvals=list(range(len(airlines))),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            yaxis=dict(
                title='Terminal',
                ticktext=terminals,
                tickvals=list(range(len(terminals))),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            zaxis=dict(
                title='Delay (min)',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        height=350,
        showlegend=False
    )
    
    return fig

# =========================
# ADVANCED KPI CARDS
# =========================
def create_advanced_kpi_card(icon, value, label, trend=None, color='#0ea5e9'):
    trend_icon = "↗️" if trend == "up" else "↘️" if trend == "down" else "➡️"
    trend_color = "#10b981" if trend == "up" else "#ef4444" if trend == "down" else "#94a3b8"
    
    return html.Div(className="kpi-card-3d", style={
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "center",
        "alignItems": "center",
        "cursor": "pointer"
    }, children=[
        html.Div(style={
            "fontSize": "32px",
            "marginBottom": "12px",
            "color": color
        }, children=icon),
        
        html.Div(className="kpi-value-3d", children=value),
        
        html.Div(style={
            "display": "flex",
            "alignItems": "center",
            "gap": "8px",
            "marginTop": "4px"
        }, children=[
            html.Div(style={
                "fontSize": "13px",
                "fontWeight": "600",
                "color": "#94a3b8",
                "textTransform": "uppercase",
                "letterSpacing": "1px"
            }, children=label),
            
            html.Div(style={
                "color": trend_color,
                "fontSize": "14px",
                "fontWeight": "700"
            }, children=trend_icon) if trend else None
        ])
    ])

# =========================
# PAGE 1: 3D DASHBOARD LAYOUT
# =========================
def dashboard_3d_layout():
    return html.Div([
        get_3d_navbar(),
        
        html.Div(className="container-3d", children=[
            html.Div(className="glass-card", style={
                "padding": "30px",
                "marginBottom": "30px",
                "background": "linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(20, 184, 166, 0.05))"
            }, children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                    html.Div(children=[
                        html.H1("✈️ AERO AI DASHBOARD", style={
                            "fontSize": "28px",
                            "fontWeight": "800",
                            "background": "linear-gradient(135deg, #0ea5e9, #14b8a6)",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                            "backgroundClip": "text",
                            "marginBottom": "8px"
                        }),
                        html.P("Real-time Flight Operations Intelligence Platform", style={
                            "color": "#94a3b8",
                            "fontSize": "16px"
                        })
                    ]),
                    
                    html.Div(style={"display": "flex", "gap": "15px"}, children=[
                        html.Div(style={"width": "200px"}, children=[
                            html.Label("TERMINAL", style={
                                "fontSize": "11px",
                                "fontWeight": "600",
                                "color": "#94a3b8",
                                "textTransform": "uppercase",
                                "letterSpacing": "1px",
                                "marginBottom": "6px"
                            }),
                            dcc.Dropdown(
                                id="global_terminal",
                                options=[{"label": "ALL TERMINALS", "value": "all"}] + 
                                        [{"label": str(t), "value": t} for t in sorted(df_flights["terminal"].unique())] if not df_flights.empty else [],
                                value="all",
                                className="input-3d",
                                clearable=False
                            )
                        ]),
                        html.Div(style={"width": "200px"}, children=[
                            html.Label("AIRLINE", style={
                                "fontSize": "11px",
                                "fontWeight": "600",
                                "color": "#94a3b8",
                                "textTransform": "uppercase",
                                "letterSpacing": "1px",
                                "marginBottom": "6px"
                            }),
                            dcc.Dropdown(
                                id="global_airline",
                                options=[{"label": "ALL AIRLINES", "value": "all"}] + 
                                        [{"label": str(a), "value": a} for a in sorted(df_flights["airline"].unique())] if not df_flights.empty else [],
                                value="all" if df_flights.empty else df_flights["airline"].unique()[0],
                                className="input-3d",
                                clearable=False
                            )
                        ])
                    ])
                ])
            ]),
            
            html.Div(id="kpi_cards", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "20px",
                "marginBottom": "30px"
            }),
            
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
                html.Div(className="glass-card", style={"padding": "20px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}, children=[
                        html.H3("3D Flight Patterns", style={
                            "fontSize": "18px",
                            "fontWeight": "600",
                            "color": "#e2e8f0"
                        }),
                        html.I(className="fas fa-cube", style={"color": "#0ea5e9", "fontSize": "20px"})
                    ]),
                    dcc.Graph(figure=create_3d_scatter(), style={"height": "300px"})
                ]),
                
                html.Div(className="glass-card", style={"padding": "20px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}, children=[
                        html.H3("Delay Surface Analysis", style={
                            "fontSize": "18px",
                            "fontWeight": "600",
                            "color": "#e2e8f0"
                        }),
                        html.I(className="fas fa-chart-area", style={"color": "#14b8a6", "fontSize": "20px"})
                    ]),
                    dcc.Graph(figure=create_3d_surface(), style={"height": "300px"})
                ])
            ]),
            
            html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "20px"}, children=[
                html.Div(className="glass-card", style={"padding": "20px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}, children=[
                        html.H3("Airline Performance 3D", style={
                            "fontSize": "18px",
                            "fontWeight": "600",
                            "color": "#e2e8f0"
                        }),
                        html.I(className="fas fa-chart-bar", style={"color": "#8b5cf6", "fontSize": "20px"})
                    ]),
                    dcc.Graph(figure=create_3d_bar(), style={"height": "350px"})
                ]),
                
                html.Div(className="glass-card", style={"padding": "20px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}, children=[
                        html.H3("Risk Distribution", style={
                            "fontSize": "18px",
                            "fontWeight": "600",
                            "color": "#e2e8f0"
                        }),
                        html.I(className="fas fa-exclamation-triangle", style={"color": "#f59e0b", "fontSize": "20px"})
                    ]),
                    dcc.Graph(id="risk_distribution", style={"height": "350px"})
                ])
            ])
        ])
    ])

# =========================
# PAGE 2: AI PREDICTION 3D LAYOUT
# =========================
def delay_prediction_3d_layout():
    return html.Div([
        get_3d_navbar(),
        
        html.Div(className="container-3d", children=[
            html.Div(className="glass-card", style={"padding": "30px"}, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "15px", "marginBottom": "20px"}, children=[
                    html.I(className="fas fa-brain", style={"color": "#0ea5e9", "fontSize": "32px"}),
                    html.Div(children=[
                        html.H1("AI FLIGHT DELAY PREDICTION", style={
                            "fontSize": "24px",
                            "fontWeight": "700",
                            "background": "linear-gradient(135deg, #0ea5e9, #8b5cf6)",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                            "backgroundClip": "text"
                        }),
                        html.P("Machine Learning Model for Predictive Analytics", style={
                            "color": "#94a3b8",
                            "fontSize": "14px"
                        })
                    ])
                ]),
                
                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "20px", "marginBottom": "30px"}, children=[
                    html.Div(children=[
                        html.Label("AIRLINE", className="form-label", style={
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "color": "#94a3b8",
                            "textTransform": "uppercase",
                            "letterSpacing": "1px",
                            "marginBottom": "8px"
                        }),
                        dcc.Dropdown(
                            id="pred_airline",
                            options=[{"label": str(a), "value": a} for a in sorted(df_flights["airline"].unique())] if not df_flights.empty else [],
                            value=df_flights["airline"].unique()[0] if not df_flights.empty else None,
                            className="input-3d",
                            clearable=False
                        )
                    ]),
                    
                    html.Div(children=[
                        html.Label("TERMINAL", className="form-label", style={
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "color": "#94a3b8",
                            "textTransform": "uppercase",
                            "letterSpacing": "1px",
                            "marginBottom": "8px"
                        }),
                        dcc.Dropdown(
                            id="pred_terminal",
                            options=[{"label": str(t), "value": t} for t in sorted(df_flights["terminal"].unique())] if not df_flights.empty else [],
                            value=df_flights["terminal"].unique()[0] if not df_flights.empty else None,
                            className="input-3d",
                            clearable=False
                        )
                    ]),
                    
                    html.Div(children=[
                        html.Label("DAY OF WEEK", className="form-label", style={
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "color": "#94a3b8",
                            "textTransform": "uppercase",
                            "letterSpacing": "1px",
                            "marginBottom": "8px"
                        }),
                        dcc.Dropdown(
                            id="pred_day",
                            options=[{"label": str(d), "value": d} for d in sorted(df_flights["day_of_week"].unique())] if not df_flights.empty else [],
                            value=df_flights["day_of_week"].unique()[0] if not df_flights.empty else None,
                            className="input-3d",
                            clearable=False
                        )
                    ]),
                    
                    html.Div(children=[
                        html.Label("WEATHER", className="form-label", style={
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "color": "#94a3b8",
                            "textTransform": "uppercase",
                            "letterSpacing": "1px",
                            "marginBottom": "8px"
                        }),
                        dcc.Dropdown(
                            id="pred_weather",
                            options=[{"label": str(w), "value": w} for w in sorted(df_flights["weather"].unique())] if not df_flights.empty else [],
                            value=df_flights["weather"].unique()[0] if not df_flights.empty else None,
                            className="input-3d",
                            clearable=False
                        )
                    ])
                ]),
                
                html.Div(style={"display": "flex", "justifyContent": "center", "marginBottom": "30px"}, children=[
                    html.Button([
                        html.I(className="fas fa-bolt", style={"marginRight": "10px"}),
                        "RUN AI PREDICTION"
                    ], id="predict_btn", className="btn-3d", style={"padding": "16px 40px", "fontSize": "16px"})
                ]),
                
                html.Hr(style={"border": "1px solid rgba(255,255,255,0.1)", "margin": "30px 0"}),
                
                html.Div(id="prediction_results", style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "30px",
                    "marginBottom": "30px"
                }),
                
                html.H3("HISTORICAL ANALYSIS", style={
                    "fontSize": "18px",
                    "fontWeight": "600",
                    "color": "#e2e8f0",
                    "marginBottom": "20px"
                }),
                
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
                    html.Div(className="glass-card", style={"padding": "20px"}, children=[
                        html.H4("Delay Distribution", style={"marginBottom": "15px", "color": "#e2e8f0"}),
                        dcc.Graph(id="delay_histogram", style={"height": "300px"})
                    ]),
                    html.Div(className="glass-card", style={"padding": "20px"}, children=[
                        html.H4("Pattern by Hour", style={"marginBottom": "15px", "color": "#e2e8f0"}),
                        dcc.Graph(id="delay_by_hour", style={"height": "300px"})
                    ])
                ])
            ])
        ])
    ])

# =========================
# PAGE 3: BEST FLIGHT 3D LAYOUT
# =========================
def best_flight_3d_layout():
    return html.Div([
        get_3d_navbar(),
        
        html.Div(className="container-3d", children=[
            html.Div(className="glass-card", style={"padding": "30px"}, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "15px", "marginBottom": "20px"}, children=[
                    html.I(className="fas fa-star", style={"color": "#f59e0b", "fontSize": "32px"}),
                    html.Div(children=[
                        html.H1("OPTIMAL FLIGHT SELECTION", style={
                            "fontSize": "24px",
                            "fontWeight": "700",
                            "background": "linear-gradient(135deg, #f59e0b, #10b981)",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                            "backgroundClip": "text"
                        }),
                        html.P("Find flights with minimal delay probability", style={
                            "color": "#94a3b8",
                            "fontSize": "14px"
                        })
                    ])
                ]),
                
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "30px"}, children=[
                    html.Div(children=[
                        html.Label("AIRLINE", className="form-label", style={
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "color": "#94a3b8",
                            "textTransform": "uppercase",
                            "letterSpacing": "1px",
                            "marginBottom": "8px"
                        }),
                        dcc.Dropdown(
                            id="advisor_airline",
                            options=[{"label": "ALL AIRLINES", "value": "all"}] + 
                                    [{"label": str(a), "value": a} for a in sorted(df_flights["airline"].unique())] if not df_flights.empty else [],
                            value="all",
                            className="input-3d",
                            clearable=False
                        )
                    ]),
                    
                    html.Div(children=[
                        html.Label("DAY OF WEEK", className="form-label", style={
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "color": "#94a3b8",
                            "textTransform": "uppercase",
                            "letterSpacing": "1px",
                            "marginBottom": "8px"
                        }),
                        dcc.Dropdown(
                            id="advisor_day",
                            options=[{"label": str(d), "value": d} for d in sorted(df_flights["day_of_week"].unique())] if not df_flights.empty else [],
                            value=df_flights["day_of_week"].unique()[0] if not df_flights.empty else None,
                            className="input-3d",
                            clearable=False
                        )
                    ])
                ]),
                
                html.Div(style={"display": "flex", "justifyContent": "center", "marginBottom": "30px"}, children=[
                    html.Button([
                        html.I(className="fas fa-search", style={"marginRight": "10px"}),
                        "FIND OPTIMAL FLIGHTS"
                    ], id="find_flights_btn", className="btn-3d", style={"padding": "16px 40px", "fontSize": "16px"})
                ]),
                
                html.Hr(style={"border": "1px solid rgba(255,255,255,0.1)", "margin": "30px 0"}),
                
                html.H3("RECOMMENDED FLIGHTS", style={
                    "fontSize": "18px",
                    "fontWeight": "600",
                    "color": "#e2e8f0",
                    "marginBottom": "20px"
                }),
                
                dcc.Graph(id="best_flight_table", style={"height": "400px"})
            ])
        ])
    ])

# =========================
# PAGE 4: ANALYTICS 3D LAYOUT
# =========================
def analytics_3d_layout():
    return html.Div([
        get_3d_navbar(),
        
        html.Div(className="container-3d", children=[
            html.Div(className="glass-card", style={
                "padding": "30px",
                "marginBottom": "30px",
                "background": "linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(20, 184, 166, 0.05))"
            }, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "15px", "marginBottom": "30px"}, children=[
                    html.I(className="fas fa-chart-bar", style={"color": "#8b5cf6", "fontSize": "32px"}),
                    html.Div(children=[
                        html.H1("ADVANCED ANALYTICS", style={
                            "fontSize": "24px",
                            "fontWeight": "700",
                            "background": "linear-gradient(135deg, #8b5cf6, #14b8a6)",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                            "backgroundClip": "text"
                        }),
                        html.P("Deep Dive Analysis & Insights", style={
                            "color": "#94a3b8",
                            "fontSize": "14px"
                        })
                    ])
                ]),
                
                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "20px"}, children=[
                    html.Div(className="glass-card", style={"padding": "20px"}, children=[
                        html.H4("Weather Impact", style={"marginBottom": "15px", "color": "#e2e8f0"}),
                        dcc.Graph(id="weather_impact", style={"height": "300px"})
                    ]),
                    html.Div(className="glass-card", style={"padding": "20px"}, children=[
                        html.H4("Hourly Patterns", style={"marginBottom": "15px", "color": "#e2e8f0"}),
                        dcc.Graph(id="hourly_pattern", style={"height": "300px"})
                    ]),
                    html.Div(className="glass-card", style={"padding": "20px"}, children=[
                        html.H4("Terminal Performance", style={"marginBottom": "15px", "color": "#e2e8f0"}),
                        dcc.Graph(id="terminal_comparison", style={"height": "300px"})
                    ]),
                    html.Div(className="glass-card", style={"padding": "20px"}, children=[
                        html.H4("Weekly Trends", style={"marginBottom": "15px", "color": "#e2e8f0"}),
                        dcc.Graph(id="day_of_week_analysis", style={"height": "300px"})
                    ])
                ])
            ])
        ])
    ])

# =========================
# MAIN LAYOUT WITH ROUTING
# =========================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == "/prediction":
        return delay_prediction_3d_layout()
    elif pathname == "/best-flight":
        return best_flight_3d_layout()
    elif pathname == "/analytics":
        return analytics_3d_layout()
    return dashboard_3d_layout()

# =========================
# DASHBOARD CALLBACKS
# =========================

@app.callback(
    Output("kpi_cards", "children"),
    Output("risk_distribution", "figure"),
    Input("global_terminal", "value"),
    Input("global_airline", "value")
)
def update_3d_dashboard(terminal, airline):
    df = df_flights.copy()
    if terminal != "all":
        df = df[df["terminal"] == terminal]
    if airline != "all":
        df = df[df["airline"] == airline]
    
    total_flights = len(df)
    avg_delay = round(df["delay_minutes"].mean(), 1) if len(df) > 0 else 0
    on_time_rate = round((df["delay_minutes"] <= 15).sum() / len(df) * 100, 1) if len(df) > 0 else 0
    high_risk = (df["delay_minutes"] > 30).sum() if len(df) > 0 else 0
    
    kpi_cards = [
        create_advanced_kpi_card("✈️", f"{total_flights:,}", "Total Flights", "up", "#0ea5e9"),
        create_advanced_kpi_card("⏱️", f"{avg_delay}", "Avg Delay", "down" if avg_delay < 20 else "up", "#f59e0b"),
        create_advanced_kpi_card("✅", f"{on_time_rate}%", "On-Time", "up" if on_time_rate > 80 else "down", "#10b981"),
        create_advanced_kpi_card("⚠️", f"{high_risk}", "High Risk", "down" if high_risk < 5 else "up", "#ef4444")
    ]
    
    if 'delay_risk' in df.columns and len(df) > 0:
        risk_counts = df['delay_risk'].value_counts().reset_index()
        risk_dist = px.pie(risk_counts, values='count', names='delay_risk',
                          hole=0.4,
                          color='delay_risk',
                          color_discrete_map={
                              'Low': '#10b981',
                              'Medium': '#f59e0b',
                              'High': '#ef4444'
                          })
    else:
        risk_dist = px.pie(values=[1], names=['No Data'])
    
    risk_dist.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            font=dict(size=12)
        ),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    risk_dist.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#1e293b', width=2))
    )
    
    return kpi_cards, risk_dist

# =========================
# PREDICTION CALLBACKS
# =========================
@app.callback(
    Output("prediction_results", "children"),
    Output("delay_histogram", "figure"),
    Output("delay_by_hour", "figure"),
    Input("predict_btn", "n_clicks"),
    State("pred_airline", "value"),
    State("pred_terminal", "value"),
    State("pred_day", "value"),
    State("pred_weather", "value"),
    prevent_initial_call=True
)
def predict_delay_3d(n_clicks, airline, terminal, day, weather):
    if None in [airline, terminal, day, weather]:
        return html.Div("⚠️ Please select all input values"), go.Figure(), go.Figure()
    
    if model is None:
        return html.Div("❌ Model not available"), go.Figure(), go.Figure()
    
    X = pd.DataFrame([{
        "airline": airline,
        "terminal": terminal,
        "day_of_week": day,
        "weather": weather
    }])
    
    try:
        pred_minutes = float(model.predict(X)[0])
        if pred_minutes < 15:
            prob = 25
            risk_level = "LOW"
            risk_color = "#10b981"
        elif pred_minutes < 30:
            prob = 55
            risk_level = "MEDIUM"
            risk_color = "#f59e0b"
        else:
            prob = 85
            risk_level = "HIGH"
            risk_color = "#ef4444"
        
        prediction_output = html.Div([
            html.Div(f"Predicted Delay: {int(pred_minutes)} minutes", 
                    style={"fontSize": "20px", "fontWeight": "600", "color": "#e2e8f0", "marginBottom": "10px"}),
            html.Div([
                "Risk Level: ",
                html.Span(risk_level, 
                         style={
                             "marginLeft": "10px",
                             "padding": "6px 15px",
                             "background": risk_color,
                             "color": "white",
                             "borderRadius": "20px",
                             "fontWeight": "700",
                             "fontSize": "14px"
                         })
            ])
        ])
        
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={"text": "Delay Probability", "font": {"size": 16, "color": "#e2e8f0"}},
            number={"suffix": "%", "font": {"size": 36, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": risk_color},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 2,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.3)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.3)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "#e2e8f0", "width": 3},
                    "thickness": 0.8,
                    "value": prob
                }
            }
        ))
        
        gauge.update_layout(
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        results = html.Div([
            html.Div(className="glass-card", style={"padding": "20px"}, children=[
                html.H4("Prediction Result", style={"marginBottom": "15px", "color": "#e2e8f0"}),
                prediction_output
            ]),
            html.Div(className="glass-card", style={"padding": "20px"}, children=[
                html.H4("Risk Assessment", style={"marginBottom": "15px", "color": "#e2e8f0"}),
                dcc.Graph(figure=gauge, style={"height": "200px"})
            ])
        ])
        
    except Exception as e:
        print(f"Prediction error: {e}")
        results = html.Div(className="glass-card", style={"padding": "20px"}, children=[
            html.H4("Prediction Error", style={"color": "#ef4444"}),
            html.P("Could not generate prediction. Please try again.", style={"color": "#94a3b8"})
        ])
    
    similar_flights = df_flights[
        (df_flights["airline"] == airline) &
        (df_flights["weather"] == weather)
    ]
    
    if len(similar_flights) > 0:
        hist = px.histogram(similar_flights, x="delay_minutes", nbins=20,
                           color_discrete_sequence=['#0ea5e9'])
        by_hour = px.box(similar_flights, x="departure_hour", y="delay_minutes",
                        color_discrete_sequence=['#0ea5e9'])
    else:
        hist = px.histogram()
        by_hour = px.box()
    
    for fig, title in [(hist, "Delay Distribution"), (by_hour, "Delay by Hour")]:
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            title=title,
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)'
            ),
            margin=dict(l=40, r=20, t=40, b=40)
        )
    
    return results, hist, by_hour

# =========================
# BEST FLIGHT CALLBACKS
# =========================
@app.callback(
    Output("best_flight_table", "figure"),
    Input("find_flights_btn", "n_clicks"),
    State("advisor_airline", "value"),
    State("advisor_day", "value"),
    prevent_initial_call=True
)
def find_best_flights_3d(n_clicks, airline, day):
    filtered = df_flights.copy()
    if airline != "all":
        filtered = filtered[filtered["airline"] == airline]
    if day != "all":
        filtered = filtered[filtered["day_of_week"] == day]
    
    filtered = filtered.sort_values("delay_minutes").head(10)
    
    if filtered.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No flights found with these criteria",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#94a3b8", size=14)
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Airline", "Terminal", "Departure", "Delay", "Risk"],
            fill_color='#1e293b',
            align='center',
            font=dict(color='white', size=12),
            height=40
        ),
        cells=dict(
            values=[
                filtered["airline"],
                filtered["terminal"],
                filtered["departure_time"].dt.strftime("%H:%M") if 'departure_time' in filtered.columns else ["N/A"]*len(filtered),
                filtered["delay_minutes"].round(1),
                filtered["delay_risk"] if 'delay_risk' in filtered.columns else ["N/A"]*len(filtered)
            ],
            fill_color=['#1e293b', '#0f172a'],
            align='center',
            font=dict(color='#e2e8f0', size=12),
            height=35
        )
    )])
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

# =========================
# ANALYTICS CALLBACKS
# =========================
@app.callback(
    Output("weather_impact", "figure"),
    Output("hourly_pattern", "figure"),
    Output("terminal_comparison", "figure"),
    Output("day_of_week_analysis", "figure"),
    Input("url", "pathname")
)
def update_analytics_3d(pathname):
    if pathname != "/analytics":
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
    if not df_flights.empty:
        weather_fig = px.box(df_flights, x="weather", y="delay_minutes",
                           color_discrete_sequence=['#0ea5e9'])
    else:
        weather_fig = px.box()
    
    if not df_flights.empty and 'departure_hour' in df_flights.columns:
        hourly_fig = px.line(
            df_flights.groupby("departure_hour")["delay_minutes"].mean().reset_index(),
            x="departure_hour", y="delay_minutes",
            color_discrete_sequence=['#14b8a6'],
            markers=True
        )
    else:
        hourly_fig = px.line()
    
    if not df_flights.empty:
        terminal_fig = px.bar(
            df_flights.groupby("terminal")["delay_minutes"].mean().reset_index(),
            x="terminal", y="delay_minutes",
            color="delay_minutes",
            color_continuous_scale=[[0, '#0ea5e9'], [1, '#14b8a6']]
        )
    else:
        terminal_fig = px.bar()
    
    if not df_flights.empty:
        day_fig = px.bar(
            df_flights.groupby("day_of_week")["delay_minutes"].mean().reset_index(),
            x="day_of_week", y="delay_minutes",
            color="delay_minutes",
            color_continuous_scale=[[0, '#8b5cf6'], [1, '#14b8a6']]
        )
    else:
        day_fig = px.bar()
    
    titles = ["Weather Impact", "Hourly Pattern", "Terminal Performance", "Weekly Trends"]
    
    for fig, title in zip([weather_fig, hourly_fig, terminal_fig, day_fig], titles):
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            title=title,
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)'
            ),
            margin=dict(l=50, r=20, t=40, b=50)
        )
        
        if "color_continuous_scale" in fig.to_dict():
            fig.update_layout(coloraxis_showscale=False)
    
    return weather_fig, hourly_fig, terminal_fig, day_fig

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=8050)