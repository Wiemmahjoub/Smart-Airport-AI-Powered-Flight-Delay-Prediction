import shap
import numpy as np
import pandas as pd
import joblib

def get_shap_explanation(airline_enc, weather_enc, terminal_enc,
                          hour, is_peak, is_weekend, model, feature_names):
    """Return SHAP values for a single prediction"""
    try:
        explainer   = shap.TreeExplainer(model)
        input_data  = np.array([[airline_enc, weather_enc, terminal_enc,
                                  hour, is_peak, is_weekend]])
        shap_values = explainer.shap_values(input_data)[0]

        explanation = []
        for feat, val, shap_val in zip(feature_names, input_data[0], shap_values):
            explanation.append({
                'feature':      feat,
                'value':        float(val),
                'shap_value':   float(shap_val),
                'impact':       'increases delay' if shap_val > 0 else 'reduces delay',
                'magnitude':    abs(float(shap_val))
            })

        # Sort by magnitude
        explanation.sort(key=lambda x: x['magnitude'], reverse=True)
        return explanation

    except Exception as e:
        print(f"SHAP error: {e}")
        return []


def format_explanation(explanation, le_airline=None,
                        le_weather=None, le_terminal=None):
    """Convert raw SHAP output to human-readable sentences"""
    readable = []
    label_map = {
        'airline_enc':      'Airline',
        'weather_enc':      'Weather',
        'terminal_enc':     'Terminal',
        'departure_hour':   'Hour of day',
        'is_peak_hour':     'Peak hour',
        'is_weekend':       'Weekend'
    }
    for item in explanation[:4]:  # Top 4 factors only
        feat  = label_map.get(item['feature'], item['feature'])
        mins  = item['shap_value']
        arrow = "▲" if mins > 0 else "▼"
        color = "#9b4a4a" if mins > 0 else "#3d7a5c"
        readable.append({
            'label':  feat,
            'text':   f"{arrow} {abs(mins):.1f} min",
            'detail': item['impact'],
            'color':  color
        })
    return readable