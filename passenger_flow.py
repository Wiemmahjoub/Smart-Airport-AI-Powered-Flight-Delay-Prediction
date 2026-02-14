# passenger_flow.py - Add this to your folder
import pandas as pd
import plotly.express as px

class PassengerFlowAnalyzer:
    def __init__(self):
        try:
            self.df = pd.read_csv('data/processed/passenger_flow_clean.csv')
        except:
            # Create sample data if file doesn't exist
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample passenger flow data"""
        import numpy as np
        hours = list(range(24))
        gates = [f'G{i}' for i in range(1, 11)]
        
        data = []
        for hour in hours:
            for gate in gates:
                # Peak at 8-10 AM and 5-7 PM
                if hour in [8, 9, 10, 17, 18, 19]:
                    passengers = np.random.randint(50, 200)
                else:
                    passengers = np.random.randint(5, 50)
                
                data.append({
                    'hour': hour,
                    'gate': gate,
                    'passenger_count': passengers
                })
        
        self.df = pd.DataFrame(data)
    
    def get_heatmap(self):
        """Generate passenger flow heatmap"""
        return px.density_heatmap(
            self.df,
            x='hour',
            y='gate',
            z='passenger_count',
            title='Passenger Flow Heatmap',
            color_continuous_scale='Viridis'
        )
    
    def get_peak_hours(self):
        """Identify peak congestion hours"""
        hourly_totals = self.df.groupby('hour')['passenger_count'].sum()
        peak_hours = hourly_totals.nlargest(3).index.tolist()
        return peak_hours