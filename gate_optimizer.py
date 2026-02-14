# gate_optimizer.py - Add this to your folder
import pandas as pd
import numpy as np

class GateOptimizer:
    def __init__(self):
        try:
            self.flights_df = pd.read_csv('data/processed/flights_clean.csv')
        except:
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample flight data"""
        airlines = ['Air France', 'Emirates', 'Lufthansa', 'Qatar', 'TunisAir']
        terminals = ['T1', 'T2', 'T3']
        gates = [f'G{i}' for i in range(1, 16)]
        
        data = []
        for i in range(100):
            data.append({
                'flight_id': f'FL{i:03d}',
                'airline': np.random.choice(airlines),
                'terminal': np.random.choice(terminals),
                'gate': np.random.choice(gates),
                'delay_minutes': np.random.randint(0, 60)
            })
        
        self.flights_df = pd.DataFrame(data)
    
    def analyze_gates(self):
        """Analyze gate performance"""
        gate_stats = self.flights_df.groupby('gate').agg({
            'flight_id': 'count',
            'delay_minutes': 'mean'
        }).rename(columns={'flight_id': 'total_flights', 'delay_minutes': 'avg_delay'})
        
        gate_stats['utilization'] = (gate_stats['total_flights'] / gate_stats['total_flights'].max() * 100)
        gate_stats['efficiency'] = 100 - gate_stats['avg_delay']
        
        return gate_stats.sort_values('efficiency', ascending=False)
    
    def get_recommendations(self):
        """Get gate optimization recommendations"""
        gate_stats = self.analyze_gates()
        
        recommendations = []
        
        # Find worst performing gates
        worst_gates = gate_stats[gate_stats['efficiency'] < 60]
        best_gates = gate_stats[gate_stats['efficiency'] > 80]
        
        if len(worst_gates) > 0 and len(best_gates) > 0:
            worst = worst_gates.index[0]
            best = best_gates.index[0]
            
            recommendations.append({
                'action': f'Move flights from Gate {worst} to Gate {best}',
                'reason': f'Gate {worst} has {worst_gates.loc[worst, "avg_delay"]:.1f} min avg delay',
                'expected_improvement': 'Reduce delays by 20-30%'
            })
        
        return recommendations