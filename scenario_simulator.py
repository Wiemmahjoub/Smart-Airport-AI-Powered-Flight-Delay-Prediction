import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class ScenarioSimulator:
    """
    Airport Digital Twin — Scenario Simulation Engine.
    Simulates cascade effects of operational scenarios.
    """

    GATE_CAPACITY    = 3   # max flights per gate per hour
    TERMINAL_GATES   = {'T1': 8, 'T2': 6, 'T3': 5}
    TURNAROUND_TIME  = 45  # minutes

    WEATHER_IMPACT = {
        'Clear':  {'delay_multiplier': 1.0,  'cancel_prob': 0.01, 'cascade_factor': 1.0},
        'Cloudy': {'delay_multiplier': 1.15, 'cancel_prob': 0.03, 'cascade_factor': 1.1},
        'Rain':   {'delay_multiplier': 1.40, 'cancel_prob': 0.08, 'cascade_factor': 1.3},
        'Storm':  {'delay_multiplier': 1.85, 'cancel_prob': 0.22, 'cascade_factor': 1.8},
    }

    def simulate(self, terminal, weather, flight_count, start_hour, end_hour,
                 base_delay=25.0, df_flights=None):
        """
        Run a full scenario simulation.

        Returns a detailed breakdown of predicted operational impact.
        """
        weather_cfg  = self.WEATHER_IMPACT.get(weather, self.WEATHER_IMPACT['Clear'])
        gate_count   = self.TERMINAL_GATES.get(terminal, 6)
        window_hours = max(1, end_hour - start_hour)
        capacity     = gate_count * self.GATE_CAPACITY * window_hours

        # ── Real base delay from data ──────────────────────────
        if df_flights is not None and not df_flights.empty:
            t_data = df_flights[df_flights['terminal'] == terminal]
            if not t_data.empty:
                base_delay = float(t_data['delay_minutes'].mean())

        # ── Core metrics ───────────────────────────────────────
        adjusted_delay      = base_delay * weather_cfg['delay_multiplier']
        overload_ratio      = flight_count / max(capacity, 1)
        overload_pct        = min(overload_ratio * 100, 100)
        gate_overload_prob  = min(overload_ratio * 85, 99)

        # Cascade delays — each overloaded gate creates downstream delays
        cascade_delay = 0.0
        if overload_ratio > 0.8:
            excess         = (overload_ratio - 0.8) * flight_count
            cascade_delay  = excess * self.TURNAROUND_TIME * weather_cfg['cascade_factor'] / max(gate_count, 1)

        total_delay          = adjusted_delay + cascade_delay
        total_delay_minutes  = round(total_delay * flight_count, 0)
        cancellations        = round(flight_count * weather_cfg['cancel_prob'])
        diverted             = round(flight_count * weather_cfg['cancel_prob'] * 0.4)
        affected_passengers  = flight_count * 180  # avg 180 pax per flight

        # ── Risk level ────────────────────────────────────────
        if total_delay > 60 or overload_ratio > 1.2:
            risk_level = "CRITICAL"
        elif total_delay > 35 or overload_ratio > 0.9:
            risk_level = "HIGH"
        elif total_delay > 20 or overload_ratio > 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # ── Hourly breakdown ──────────────────────────────────
        hourly = []
        flights_per_hour = flight_count / max(window_hours, 1)
        for h in range(start_hour, end_hour):
            hour_overload = flights_per_hour / (gate_count * self.GATE_CAPACITY)
            hour_delay    = adjusted_delay * (1 + max(0, hour_overload - 0.7) * 0.5)
            hourly.append({
                'hour':          h,
                'flights':       round(flights_per_hour, 1),
                'avg_delay':     round(hour_delay, 1),
                'gate_load_pct': round(min(hour_overload * 100, 100), 1),
                'status':        'OVERLOADED' if hour_overload > 1 else
                                 'HIGH' if hour_overload > 0.8 else 'NORMAL'
            })

        # ── Recommendations ───────────────────────────────────
        recommendations = self._generate_recommendations(
            terminal, weather, overload_ratio, total_delay,
            cancellations, gate_count, flight_count, window_hours
        )

        return {
            'summary': {
                'terminal':            terminal,
                'weather':             weather,
                'flight_count':        flight_count,
                'window':              f"{start_hour}:00 – {end_hour}:00",
                'gate_count':          gate_count,
                'capacity':            capacity,
                'risk_level':          risk_level,
            },
            'metrics': {
                'avg_delay_min':         round(total_delay, 1),
                'total_delay_minutes':   int(total_delay_minutes),
                'cascade_delay_min':     round(cascade_delay, 1),
                'gate_overload_pct':     round(gate_overload_prob, 1),
                'terminal_capacity_pct': round(overload_pct, 1),
                'cancellations':         int(cancellations),
                'diversions':            int(diverted),
                'affected_passengers':   int(affected_passengers),
            },
            'hourly_breakdown': hourly,
            'recommendations':  recommendations,
            'simulated_at':     datetime.now().isoformat()
        }

    def _generate_recommendations(self, terminal, weather, overload_ratio,
                                   total_delay, cancellations, gate_count,
                                   flight_count, window_hours):
        recs = []

        if overload_ratio > 1.0:
            recs.append({
                'priority': 'CRITICAL',
                'action':   f'Redistribute {round((overload_ratio - 1) * flight_count)} flights to T2/T3',
                'impact':   'Reduces gate overload to acceptable levels'
            })

        if weather in ['Rain', 'Storm']:
            recs.append({
                'priority': 'HIGH',
                'action':   f'Activate weather contingency protocol for {terminal}',
                'impact':   f'Mitigate {weather} impact — pre-position de-icing/ground crew'
            })

        if total_delay > 40:
            recs.append({
                'priority': 'HIGH',
                'action':   'Issue proactive passenger delay notifications',
                'impact':   f'Avg delay {round(total_delay)} min expected — notify 2h in advance'
            })

        if overload_ratio > 0.8:
            extra_gates = max(1, round((overload_ratio - 0.8) * gate_count))
            recs.append({
                'priority': 'MEDIUM',
                'action':   f'Open {extra_gates} additional gate(s) in {terminal}',
                'impact':   'Increase throughput capacity by 15–25%'
            })

        if cancellations > 0:
            recs.append({
                'priority': 'MEDIUM',
                'action':   f'Pre-arrange rebooking for ~{cancellations} expected cancellations',
                'impact':   'Reduce passenger disruption and complaint volume'
            })

        if window_hours <= 2 and flight_count > 8:
            recs.append({
                'priority': 'MEDIUM',
                'action':   'Stagger departure slots by 10–15 min intervals',
                'impact':   'Smooth peak load and reduce cascade delays'
            })

        if not recs:
            recs.append({
                'priority': 'LOW',
                'action':   'Normal operations — standard monitoring sufficient',
                'impact':   'No immediate action required'
            })

        return recs
