"""
Module 8: Flask API for Smart Energy Consumption Analysis
Provides REST API endpoints for energy predictions and device insights.
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Setup paths - use the directory where app.py is located
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

# Global variables for loaded models and data
models = {}
data_cache = {}


def load_models():
    """Load trained models into memory."""
    global models
    
    # Load Linear Regression model
    try:
        with open('models/linear_regression.pkl', 'rb') as f:
            models['linear_regression'] = pickle.load(f)
        print("✓ Linear Regression model loaded")
    except Exception as e:
        print(f"✗ Failed to load Linear Regression model: {e}")
    
    # Load LSTM model
    try:
        from tensorflow.keras.models import load_model
        models['lstm'] = load_model('models/lstm_model.keras')
        print("✓ LSTM model loaded")
    except Exception as e:
        print(f"✗ Failed to load LSTM model: {e}")


def load_data():
    """Load and cache the energy data."""
    global data_cache
    
    try:
        # Load clean hourly data
        data_cache['hourly'] = pd.read_csv('clean_energy_data.csv', index_col=0, parse_dates=True)
        
        # Load train features for reference
        data_cache['train'] = pd.read_csv('train_features.csv', index_col=0, parse_dates=True)
        data_cache['test'] = pd.read_csv('test_features.csv', index_col=0, parse_dates=True)
        
        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")


def get_device_columns():
    """Get list of device columns from the dataset."""
    device_cols = [
        'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
        'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
        'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
        'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
        'Microwave [kW]', 'Living room [kW]'
    ]
    return device_cols


def generate_smart_suggestions(device_data):
    """Generate energy-saving suggestions based on device usage patterns."""
    suggestions = []
    
    if 'hourly' not in data_cache:
        return suggestions
    
    df = data_cache['hourly']
    
    # Analyze each device
    device_cols = get_device_columns()
    
    for col in device_cols:
        if col in df.columns:
            avg_usage = df[col].mean()
            max_usage = df[col].max()
            
            device_name = col.replace(' [kW]', '')
            
            # High usage devices
            if avg_usage > 0.3:
                suggestions.append({
                    'device': device_name,
                    'type': 'high_usage',
                    'message': f'{device_name} has high average consumption. Consider using during off-peak hours.',
                    'priority': 'high',
                    'potential_savings': f'{(avg_usage * 0.15 * 24 * 30):.2f} kWh/month'
                })
            
            # Furnace optimization
            if 'Furnace' in col and avg_usage > 0.2:
                suggestions.append({
                    'device': device_name,
                    'type': 'hvac',
                    'message': f'Optimize {device_name} by lowering thermostat 2°F to save up to 5% on heating costs.',
                    'priority': 'medium',
                    'potential_savings': f'{(avg_usage * 0.05 * 24 * 30):.2f} kWh/month'
                })
            
            # Kitchen appliances
            if 'Kitchen' in col and avg_usage > 0.1:
                suggestions.append({
                    'device': device_name,
                    'type': 'kitchen',
                    'message': f'Batch cooking can reduce {device_name} usage by consolidating cooking sessions.',
                    'priority': 'low',
                    'potential_savings': f'{(avg_usage * 0.1 * 24 * 30):.2f} kWh/month'
                })
    
    # Overall suggestions
    total_usage = df['use [kW]'].mean() if 'use [kW]' in df.columns else 0
    if total_usage > 1.0:
        suggestions.insert(0, {
            'device': 'Overall',
            'type': 'general',
            'message': 'Your overall energy usage is above average. Consider an energy audit.',
            'priority': 'high',
            'potential_savings': f'{(total_usage * 0.1 * 24 * 30):.2f} kWh/month'
        })
    
    return suggestions[:5]  # Return top 5 suggestions


@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('dashboard.html')


@app.route('/api/overview')
def get_overview():
    """Get overall energy consumption overview."""
    if 'hourly' not in data_cache:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = data_cache['hourly']
    
    # Calculate statistics
    total_usage = df['use [kW]'].sum() if 'use [kW]' in df.columns else 0
    avg_usage = df['use [kW]'].mean() if 'use [kW]' in df.columns else 0
    max_usage = df['use [kW]'].max() if 'use [kW]' in df.columns else 0
    min_usage = df['use [kW]'].min() if 'use [kW]' in df.columns else 0
    
    # Solar generation
    solar_gen = df['Solar [kW]'].sum() if 'Solar [kW]' in df.columns else 0
    
    return jsonify({
        'total_usage_kwh': round(total_usage, 2),
        'avg_usage_kw': round(avg_usage, 4),
        'max_usage_kw': round(max_usage, 4),
        'min_usage_kw': round(min_usage, 4),
        'solar_generation_kwh': round(solar_gen, 2),
        'data_points': len(df),
        'date_range': {
            'start': str(df.index.min()),
            'end': str(df.index.max())
        }
    })


@app.route('/api/devices')
def get_device_data():
    """Get device-wise consumption data."""
    if 'hourly' not in data_cache:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = data_cache['hourly']
    device_cols = get_device_columns()
    
    devices = []
    for col in device_cols:
        if col in df.columns:
            device_name = col.replace(' [kW]', '')
            devices.append({
                'name': device_name,
                'avg_usage': round(df[col].mean(), 4),
                'max_usage': round(df[col].max(), 4),
                'total_usage': round(df[col].sum(), 2),
                'percentage': round((df[col].sum() / df['use [kW]'].sum()) * 100, 2) if df['use [kW]'].sum() > 0 else 0
            })
    
    # Sort by total usage
    devices.sort(key=lambda x: x['total_usage'], reverse=True)
    
    return jsonify({'devices': devices})


@app.route('/api/hourly')
def get_hourly_data():
    """Get hourly consumption pattern."""
    if 'hourly' not in data_cache:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = data_cache['hourly'].copy()
    df['hour'] = df.index.hour
    
    hourly_pattern = df.groupby('hour')['use [kW]'].mean().to_dict()
    
    return jsonify({
        'labels': list(range(24)),
        'data': [round(hourly_pattern.get(h, 0), 4) for h in range(24)]
    })


@app.route('/api/daily')
def get_daily_data():
    """Get daily consumption trend."""
    if 'hourly' not in data_cache:
        return jsonify({'error': 'Data not loaded'}), 500
    
    df = data_cache['hourly'].copy()
    daily = df.resample('D')['use [kW]'].sum()
    
    return jsonify({
        'labels': [str(d.date()) for d in daily.index],
        'data': [round(v, 2) for v in daily.values]
    })


@app.route('/api/suggestions')
def get_suggestions():
    """Get smart energy-saving suggestions."""
    suggestions = generate_smart_suggestions(data_cache.get('hourly'))
    return jsonify({'suggestions': suggestions})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make energy consumption prediction."""
    if 'linear_regression' not in models:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json or {}
        model_type = data.get('model', 'linear_regression')
        
        # Use test data for prediction demo
        if 'test' in data_cache:
            test_df = data_cache['test']
            feature_cols = [c for c in test_df.columns if c != 'use [kW]']
            X = test_df[feature_cols].values
            y_actual = test_df['use [kW]'].values if 'use [kW]' in test_df.columns else []
            
            if model_type == 'linear_regression' and 'linear_regression' in models:
                predictions = models['linear_regression'].predict(X)
            elif model_type == 'lstm' and 'lstm' in models:
                # LSTM needs 3D input
                X_3d = X.reshape(-1, 1, X.shape[1])
                predictions = models['lstm'].predict(X_3d).flatten()
            else:
                predictions = models['linear_regression'].predict(X)
            
            return jsonify({
                'predictions': [round(p, 4) for p in predictions.tolist()],
                'actual': [round(a, 4) for a in y_actual.tolist()] if len(y_actual) > 0 else [],
                'model': model_type
            })
        
        return jsonify({'error': 'No data available for prediction'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-comparison')
def get_model_comparison():
    """Get model performance comparison data."""
    try:
        with open('results/baseline_predictions.pkl', 'rb') as f:
            baseline = pickle.load(f)
        with open('results/lstm_predictions.pkl', 'rb') as f:
            lstm = pickle.load(f)
        
        return jsonify({
            'baseline': {
                'mae': round(baseline['metrics']['mae'], 6),
                'rmse': round(baseline['metrics']['rmse'], 6),
                'r2': round(baseline['metrics']['r2'], 6)
            },
            'lstm': {
                'mae': round(lstm['metrics']['mae'], 6),
                'rmse': round(lstm['metrics']['rmse'], 6),
                'r2': round(lstm['metrics']['r2'], 6) if not np.isnan(lstm['metrics']['r2']) else 'N/A'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Smart Energy Consumption Analysis - Dashboard")
    print("=" * 50)
    
    print("\nLoading models...")
    load_models()
    
    print("\nLoading data...")
    load_data()
    
    print("\n" + "=" * 50)
    print("Starting Flask server...")
    print("Dashboard: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    
    app.run(debug=True, port=5000)
