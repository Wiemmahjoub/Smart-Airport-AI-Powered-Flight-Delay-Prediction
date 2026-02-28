import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

# =========================
# TEST DATA
# =========================
@pytest.fixture
def sample_flights():
    return pd.DataFrame({
        'flight_id':      ['FL001', 'FL002', 'FL003', 'FL004', 'FL005'],
        'airline':        ['TunisAir', 'Air France', 'Emirates', 'TunisAir', 'Lufthansa'],
        'terminal':       ['T1', 'T2', 'T3', 'T1', 'T2'],
        'gate':           ['T1-G1', 'T2-G2', 'T3-G1', 'T1-G2', 'T2-G1'],
        'delay_minutes':  [85, 20, 5, 110, 12],
        'weather':        ['Storm', 'Clear', 'Clear', 'Rain', 'Cloudy'],
        'departure_hour': [8, 14, 10, 17, 9],
        'is_peak_hour':   [1, 0, 0, 1, 1],
        'is_weekend':     [0, 1, 0, 0, 1],
        'delay_risk':     ['High', 'Medium', 'Low', 'High', 'Low']
    })


# =========================
# TEST: RISK SCORER
# =========================
class TestOperationalRiskScorer:

    def setup_method(self):
        from app_aoip import OperationalRiskScorer
        self.scorer = OperationalRiskScorer()

    def test_storm_gives_high_score(self):
        score = self.scorer.calculate_risk_score('T1', 8, 'Storm')
        assert score > 60, f"Storm at peak hour should give high risk, got {score}"

    def test_clear_weather_gives_low_score(self):
        score = self.scorer.calculate_risk_score('T1', 14, 'Clear')
        assert score < 60, f"Clear weather off-peak should give lower risk, got {score}"

    def test_score_never_exceeds_100(self):
        score = self.scorer.calculate_risk_score('T1', 8, 'Storm')
        assert score <= 100, f"Score should never exceed 100, got {score}"

    def test_score_never_below_zero(self):
        score = self.scorer.calculate_risk_score('T1', 3, 'Clear')
        assert score >= 0, f"Score should never be negative, got {score}"

    def test_storm_riskier_than_clear(self):
        storm = self.scorer.calculate_risk_score('T1', 12, 'Storm')
        clear = self.scorer.calculate_risk_score('T1', 12, 'Clear')
        assert storm > clear, "Storm should always be riskier than Clear"

    def test_peak_hour_riskier_than_offpeak(self):
        peak    = self.scorer.calculate_risk_score('T1', 8,  'Clear')
        offpeak = self.scorer.calculate_risk_score('T1', 14, 'Clear')
        assert peak > offpeak, "Peak hour should be riskier than off-peak"


# =========================
# TEST: DELAY CATEGORIZATION
# =========================
class TestDelayCategorization:

    def test_low_risk_threshold(self):
        assert self._categorize(10) == 'Low'
        assert self._categorize(15) == 'Low'

    def test_medium_risk_threshold(self):
        assert self._categorize(16) == 'Medium'
        assert self._categorize(30) == 'Medium'

    def test_high_risk_threshold(self):
        assert self._categorize(31)  == 'High'
        assert self._categorize(120) == 'High'

    def test_zero_delay_is_low(self):
        assert self._categorize(0) == 'Low'

    def _categorize(self, delay):
        if delay <= 15:  return 'Low'
        elif delay <= 30: return 'Medium'
        else:             return 'High'


# =========================
# TEST: PASSENGER FLOW
# =========================
class TestPassengerFlowIntelligence:

    def setup_method(self):
        from app_aoip import PassengerFlowIntelligence
        self.pfi = PassengerFlowIntelligence()

    def test_data_loads(self):
        assert self.pfi.data is not None
        assert len(self.pfi.data) > 0

    def test_data_has_required_columns(self):
        required = ['hour', 'terminal', 'gate', 'passenger_count']
        for col in required:
            assert col in self.pfi.data.columns, f"Missing column: {col}"

    def test_heatmap_returns_figure(self):
        import plotly.graph_objects as go
        fig = self.pfi.get_heatmap('All')
        assert fig is not None

    def test_heatmap_filters_terminal(self):
        fig_all = self.pfi.get_heatmap('All')
        fig_t1  = self.pfi.get_heatmap('T1')
        assert fig_all is not None
        assert fig_t1  is not None

    def test_hour_range_valid(self):
        hours = self.pfi.data['hour'].unique()
        assert all(0 <= h <= 23 for h in hours), "Hours should be 0-23"

    def test_passenger_count_positive(self):
        assert (self.pfi.data['passenger_count'] >= 0).all(), "Passenger counts should be positive"


# =========================
# TEST: GATE OPTIMIZATION
# =========================
class TestGateOptimizationEngine:

    def setup_method(self):
        from app_aoip import GateOptimizationEngine
        self.engine = GateOptimizationEngine()

    def test_suggestions_not_empty(self):
        suggestions = self.engine.get_optimization_suggestions()
        assert len(suggestions) > 0

    def test_suggestions_have_required_keys(self):
        suggestions = self.engine.get_optimization_suggestions()
        for s in suggestions:
            assert 'action'   in s
            assert 'reason'   in s
            assert 'impact'   in s
            assert 'priority' in s

    def test_priority_values_valid(self):
        suggestions = self.engine.get_optimization_suggestions()
        valid = {'HIGH', 'MEDIUM', 'LOW'}
        for s in suggestions:
            assert s['priority'] in valid, f"Invalid priority: {s['priority']}"

    def test_gate_performance_has_utilization(self):
        perf = self.engine.analyze_gate_performance()
        assert 'utilization' in perf.columns
        assert 'efficiency'  in perf.columns

    def test_utilization_between_0_and_100(self):
        perf = self.engine.analyze_gate_performance()
        assert (perf['utilization'] >= 0).all()
        assert (perf['utilization'] <= 100).all()


# =========================
# TEST: ML MODEL (if loaded)
# =========================
class TestMLModel:

    def test_model_file_exists(self):
        assert os.path.exists("model/delay_model.pkl"), "delay_model.pkl not found"

    def test_encoders_exist(self):
        assert os.path.exists("model/le_airline.pkl"),  "le_airline.pkl not found"
        assert os.path.exists("model/le_weather.pkl"),  "le_weather.pkl not found"
        assert os.path.exists("model/le_terminal.pkl"), "le_terminal.pkl not found"

    def test_model_predicts(self):
        import joblib
        model      = joblib.load("model/delay_model.pkl")
        le_airline = joblib.load("model/le_airline.pkl")
        features   = np.array([[0, 0, 0, 8, 1, 0]])
        prediction = model.predict(features)
        assert len(prediction) == 1
        assert prediction[0] >= 0, "Prediction should be non-negative"

    def test_weather_model_predicts_proba(self):
        import joblib
        if not os.path.exists("model/weather_delay_model.pkl"):
            pytest.skip("weather_delay_model.pkl not found")
        model = joblib.load("model/weather_delay_model.pkl")
        proba = model.predict_proba(np.array([[0, 8, 1, 0, 0]]))
        assert proba.shape[1] == 2, "Should output 2 probabilities"
        assert abs(proba[0].sum() - 1.0) < 0.001, "Probabilities should sum to 1"
