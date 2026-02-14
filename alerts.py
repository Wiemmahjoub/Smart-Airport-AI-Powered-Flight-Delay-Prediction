# alerts.py - Add to your aoip folder
import pandas as pd
from datetime import datetime, timedelta
import json

class AOIPAlertSystem:
    """Enterprise-grade alert system for airport operations"""
    
    def __init__(self):
        self.alerts_log = []
        self.alert_rules = self.load_alert_rules()
        
    def load_alert_rules(self):
        """Define alert conditions and thresholds"""
        return {
            'delays': {
                'threshold': 30,  # minutes
                'multiple_flights': 3,
                'priority': 'HIGH'
            },
            'congestion': {
                'threshold': 75,  # percentage
                'duration': 2,    # hours
                'priority': 'MEDIUM'
            },
            'gate_conflict': {
                'time_window': 30,  # minutes
                'priority': 'CRITICAL'
            },
            'weather_impact': {
                'conditions': ['Storm', 'Heavy Rain'],
                'priority': 'HIGH'
            }
        }
    
    def check_all_alerts(self, flights_df, passenger_data, weather="Clear"):
        """Run all alert checks"""
        alerts = []
        
        alerts.extend(self.check_delays(flights_df))
        alerts.extend(self.check_congestion(passenger_data))
        alerts.extend(self.check_weather_impact(weather))
        alerts.extend(self.check_resource_allocation(flights_df))
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        alerts.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return alerts
    
    def check_delays(self, flights_df):
        """Check for significant delays"""
        alerts = []
        
        if flights_df.empty:
            return alerts
        
        # Check for flights delayed > 30 minutes
        high_delays = flights_df[flights_df['delay_minutes'] > self.alert_rules['delays']['threshold']]
        
        if len(high_delays) >= self.alert_rules['delays']['multiple_flights']:
            airlines = high_delays['airline'].unique()[:3]
            alerts.append({
                'id': f"DELAY_{datetime.now().strftime('%H%M')}",
                'type': 'DELAY',
                'priority': self.alert_rules['delays']['priority'],
                'title': f'Multiple High Delays Detected',
                'message': f'{len(high_delays)} flights delayed >{self.alert_rules["delays"]["threshold"]}min',
                'details': f'Airlines affected: {", ".join(airlines)}',
                'timestamp': datetime.now().isoformat(),
                'actions': [
                    'Notify affected airlines',
                    'Update passenger information',
                    'Consider schedule adjustments'
                ]
            })
        
        return alerts
    
    def check_congestion(self, passenger_data):
        """Check for passenger congestion"""
        alerts = []
        
        if passenger_data.empty:
            return alerts
        
        # Simplified congestion check
        current_hour = datetime.now().hour
        hour_data = passenger_data[passenger_data['hour'] == current_hour]
        
        if not hour_data.empty:
            max_congestion = hour_data['passenger_count'].max()
            avg_congestion = hour_data['passenger_count'].mean()
            
            if avg_congestion > 100:  # Example threshold
                alerts.append({
                    'id': f"CONGEST_{datetime.now().strftime('%H%M')}",
                    'type': 'CONGESTION',
                    'priority': 'MEDIUM',
                    'title': 'High Passenger Congestion',
                    'message': f'Average congestion: {avg_congestion:.0f} passengers/hour',
                    'details': f'Peak at {max_congestion} passengers',
                    'timestamp': datetime.now().isoformat(),
                    'actions': [
                        'Deploy additional staff',
                        'Open extra security lanes',
                        'Monitor queue lengths'
                    ]
                })
        
        return alerts
    
    def check_weather_impact(self, weather):
        """Check weather conditions"""
        alerts = []
        
        if weather in self.alert_rules['weather_impact']['conditions']:
            alerts.append({
                'id': f"WEATHER_{datetime.now().strftime('%H%M')}",
                'type': 'WEATHER',
                'priority': self.alert_rules['weather_impact']['priority'],
                'title': f'Adverse Weather Alert: {weather}',
                'message': 'Weather conditions may impact operations',
                'details': 'Increased risk of delays and cancellations',
                'timestamp': datetime.now().isoformat(),
                'actions': [
                    'Activate weather contingency plan',
                    'Coordinate with ground services',
                    'Update flight schedules'
                ]
            })
        
        return alerts
    
    def check_resource_allocation(self, flights_df):
        """Check resource utilization"""
        alerts = []
        
        # Example: Check gate utilization
        if not flights_df.empty and 'gate' in flights_df.columns:
            gate_counts = flights_df['gate'].value_counts()
            if len(gate_counts) > 0:
                max_utilization = gate_counts.max()
                avg_utilization = gate_counts.mean()
                
                if max_utilization > avg_utilization * 2:  # Significantly higher
                    busy_gate = gate_counts.idxmax()
                    alerts.append({
                        'id': f"GATE_{datetime.now().strftime('%H%M')}",
                        'type': 'RESOURCE',
                        'priority': 'MEDIUM',
                        'title': f'Gate {busy_gate} Overutilized',
                        'message': f'{max_utilization} flights vs average {avg_utilization:.1f}',
                        'details': 'Consider redistributing flights',
                        'timestamp': datetime.now().isoformat(),
                        'actions': [
                            f'Move some flights from {busy_gate}',
                            'Review gate assignment algorithm',
                            'Check for scheduling conflicts'
                        ]
                    })
        
        return alerts
    
    def save_alerts_to_file(self, alerts, filename='alerts_log.json'):
        """Save alerts to JSON file for persistence"""
        try:
            with open(filename, 'w') as f:
                json.dump(alerts, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving alerts: {e}")
    
    def get_recent_alerts(self, hours=24):
        """Get alerts from last X hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts_log 
                        if datetime.fromisoformat(a['timestamp']) > cutoff]
        return recent_alerts
        