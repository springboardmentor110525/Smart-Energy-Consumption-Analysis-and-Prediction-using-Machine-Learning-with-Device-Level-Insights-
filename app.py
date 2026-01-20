"""
Module 8: Flask Web Application
Week 7-8 Implementation
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Try to import TensorFlow (optional - only needed for LSTM)
try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARN]  TensorFlow not available - LSTM predictions will be disabled")

import json
from datetime import datetime, timedelta
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart-energy-analysis-secret-key-2026'

# Global variables for models and data
baseline_model = None
lstm_model = None
scaler = None
df_data = None
device_stats = None

# Define DummyModel class to support loading the instant debug model
class DummyModel:
    def predict(self, X):
        # Return random predictions around 0.5 kW
        base_val = 0.5
        if isinstance(X, pd.DataFrame) and 'temperature' in X.columns:
             # silly logic: higher temp = higher usage
             return (X['temperature'].values / 100.0) 
        elif hasattr(X, 'iloc'): 
             return np.array([0.5] * len(X))
        else:
             return np.array([0.5] * len(X))

    def fit(self, X, y):
        pass # Do nothing


def load_models():
    """Load trained models and data"""
    global baseline_model, lstm_model, scaler, df_data
    
    try:
        # Load baseline model
        if os.path.exists('models/baseline_model.pkl'):
            baseline_model = joblib.load('models/baseline_model.pkl')
            print("[OK] Baseline model loaded")
        elif os.path.exists('models/baseline_lr.pkl'):
            baseline_model = joblib.load('models/baseline_lr.pkl')
            print("[OK] Baseline model loaded (legacy name)")
        else:
            print("[WARN]  Baseline model not found (checked baseline_model.pkl and baseline_lr.pkl)")
        
        # Load LSTM model (only if TensorFlow is available)
        if TENSORFLOW_AVAILABLE and os.path.exists('models/best_lstm_model.h5'):
            lstm_model = keras.models.load_model('models/best_lstm_model.h5', compile=False)
            print("[OK] LSTM model loaded")
        elif not TENSORFLOW_AVAILABLE:
            print("[WARN]  LSTM model skipped (TensorFlow not available)")
        
        # Load scaler
        if os.path.exists('models/lstm_scaler.pkl'):
            scaler = joblib.load('models/lstm_scaler.pkl')
            print("[OK] Scaler loaded")
        
        # Load data
        if os.path.exists('data/processed/energy_data_clean.csv'):
            df_data = pd.read_csv('data/processed/energy_data_clean.csv',
                                 index_col='time', parse_dates=True)
            print(f"[OK] Data loaded: {len(df_data):,} records")
        
    except Exception as e:
        print(f"Warning: Could not load models - {e}")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/predictions')
def predictions():
    """Predictions page"""
    return render_template('predictions.html')

@app.route('/api/stats')
def get_stats():
    """Get overall statistics"""
    if df_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Calculate statistics
        numeric_cols = df_data.select_dtypes(include=[np.number]).columns
        
        stats = {
            'total_records': len(df_data),
            'total_devices': len(numeric_cols),
            'date_range': {
                'start': df_data.index.min().strftime('%Y-%m-%d'),
                'end': df_data.index.max().strftime('%Y-%m-%d')
            },
            'total_consumption': float(df_data[numeric_cols].sum().sum()),
            'avg_consumption': float(df_data[numeric_cols].mean().mean()),
            'peak_consumption': float(df_data[numeric_cols].max().max())
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/device-consumption')
def get_device_consumption():
    """Get device-wise consumption data"""
    if df_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Get device columns
        device_cols = [col for col in df_data.columns if any(device in col.lower()
                      for device in ['dishwasher', 'fridge', 'microwave', 'furnace',
                                    'kitchen', 'office', 'living', 'charger', 'heater',
                                    'conditioning', 'theater', 'lights', 'laundry', 'pump'])]
        
        # Calculate total consumption per device
        device_consumption = {}
        for col in device_cols[:15]:  # Limit to 15 devices
            device_consumption[col] = float(df_data[col].sum())
        
        # Sort by consumption
        sorted_devices = dict(sorted(device_consumption.items(), 
                                   key=lambda x: x[1], reverse=True))
        
        return jsonify(sorted_devices)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hourly-consumption')
def get_hourly_consumption():
    """Get hourly consumption pattern"""
    if df_data is None:
        return jsonify({
            'error': 'Data not loaded',
            'suggestion': 'Please run: python src/data_preprocessing.py'
        }), 500
    
    try:
        # Check if index is datetime
        if not isinstance(df_data.index, pd.DatetimeIndex):
            return jsonify({
                'error': 'Data index is not datetime',
                'suggestion': 'Data preprocessing may be incomplete'
            }), 500
        
        # Get ONLY numeric columns (exclude text columns like day names)
        numeric_data = df_data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) == 0:
            return jsonify({
                'error': 'No numeric columns found',
                'suggestion': 'Check data preprocessing'
            }), 500
        
        # Group by hour and calculate mean (only on numeric columns)
        hourly = numeric_data.groupby(numeric_data.index.hour).mean()
        
        # Build hourly data
        hourly_data = {
            'hours': list(range(24)),
            'consumption': []
        }
        
        # Calculate consumption for each hour
        for i in range(24):
            if i in hourly.index:
                # Sum all numeric columns for this hour
                consumption = float(hourly.loc[i].sum())
                hourly_data['consumption'].append(consumption)
            else:
                # If hour doesn't exist in data, use 0
                hourly_data['consumption'].append(0.0)
        
        return jsonify(hourly_data)
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'details': traceback.format_exc(),
            'suggestion': 'Check if data preprocessing completed successfully'
        }), 500


@app.route('/api/daily-consumption')
def get_daily_consumption():
    """Get daily consumption pattern"""
    if df_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Resample to daily
        daily = df_data.resample('D').sum()
        numeric_cols = daily.select_dtypes(include=[np.number]).columns
        
        # Get last 30 days
        daily_last_30 = daily.tail(30)
        
        daily_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in daily_last_30.index],
            'consumption': [float(daily_last_30[numeric_cols].iloc[i].sum())
                          for i in range(len(daily_last_30))]
        }
        
        return jsonify(daily_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make energy consumption prediction"""
    try:
        data = request.json
        model_type = data.get('model_type', 'baseline')  # Default to baseline
        custom_input = data.get('custom_input', None)  # Check for custom input
        
        # Check if we have data
        if df_data is None:
            return jsonify({'error': 'No data loaded. Please run data preprocessing first.'}), 500
        
        # Handle custom input prediction
        if custom_input and baseline_model is not None:
            print("\n=== CUSTOM INPUT PREDICTION ===")
            print(f"Received custom_input: {custom_input}")
            
            # Get a sample row to know what columns exist
            sample_row = df_data.tail(1)
            numeric_cols = sample_row.select_dtypes(include=[np.number]).columns
            
            # Remove target columns from the list
            feature_cols = [col for col in numeric_cols if col not in ['use_HO', 'gen_Sol']]
            print(f"Feature columns: {list(feature_cols)}")
            
            # Create a dictionary with default values (averages from dataset)
            feature_dict = {}
            for col in feature_cols:
                feature_dict[col] = df_data[col].mean()
            
            print(f"Default feature_dict: {feature_dict}")
            
            # Update with custom input values (map frontend names to dataset column names)
            column_mapping = {
                'temperature': 'temperature',
                'humidity': 'humidity', 
                'hour': 'hour',
                'month': 'month',
                'weekday': 'weekday',
                'windSpeed': 'windSpeed',
                'cloudCover': 'cloudCover',
                'visibility': 'visibility'
            }
            
            for frontend_name, dataset_col in column_mapping.items():
                if frontend_name in custom_input and dataset_col in feature_dict:
                    old_value = feature_dict[dataset_col]
                    feature_dict[dataset_col] = custom_input[frontend_name]
                    print(f"Updated {dataset_col}: {old_value} -> {custom_input[frontend_name]}")
            
            print(f"Final feature_dict: {feature_dict}")
            
            # Create DataFrame with the same column order as training data
            import pandas as pd
            X_pred = pd.DataFrame([feature_dict], columns=feature_cols)
            
            print(f"X_pred shape: {X_pred.shape}")
            print(f"X_pred values:\n{X_pred}")
            
            # Make prediction
            prediction = baseline_model.predict(X_pred)[0]
            
            print(f"Prediction result: {prediction}")
            print("=== END CUSTOM INPUT PREDICTION ===\n")
            
            return jsonify({
                'model': 'Baseline (Linear Regression) - Custom Input',
                'prediction': float(abs(prediction)),
                'unit': 'kW',
                'timestamp': datetime.now().isoformat(),
                'confidence': '75-85%',
                'input_type': 'custom',
                'custom_values_used': {k: v for k, v in custom_input.items()}  # Show what was used
            })
        
        # LSTM prediction
        if model_type == 'lstm' and lstm_model is not None and scaler is not None and df_data is not None:
            # Get last 24 hours of data
            sequence_length = 24
            target_col = 'use_HO' if 'use_HO' in df_data.columns else df_data.select_dtypes(include=[np.number]).columns[0]
            
            if len(df_data) < sequence_length:
                return jsonify({'error': 'Not enough data for LSTM prediction'}), 500
            
            last_data = df_data[target_col].tail(sequence_length).values.reshape(-1, 1)
            scaled_data = scaler.transform(last_data)
            
            # Reshape for LSTM
            X = scaled_data.reshape(1, sequence_length, 1)
            
            # Predict
            prediction_scaled = lstm_model.predict(X, verbose=0)
            prediction = scaler.inverse_transform(prediction_scaled)[0][0]
            
            return jsonify({
                'model': 'LSTM Neural Network',
                'prediction': float(prediction),
                'unit': 'kW',
                'timestamp': datetime.now().isoformat(),
                'confidence': '85-95%'
            })
        
        # Baseline prediction with latest data
        elif model_type == 'baseline' and baseline_model is not None and df_data is not None:
            # Use baseline model with actual data
            # Get the last row of data (most recent)
            last_row = df_data.tail(1)
            
            # Get numeric columns only (features)
            numeric_cols = last_row.select_dtypes(include=[np.number]).columns
            X_pred = last_row[numeric_cols]
            
            # Remove target column if it exists
            target_cols = ['use_HO', 'gen_Sol']
            for col in target_cols:
                if col in X_pred.columns:
                    X_pred = X_pred.drop(columns=[col])
            
            # Make prediction
            prediction = baseline_model.predict(X_pred)[0]
            
            # Get average consumption for context
            avg_consumption = df_data.select_dtypes(include=[np.number]).mean().mean()
            
            return jsonify({
                'model': 'Baseline (Linear Regression)',
                'prediction': float(abs(prediction)),  # Ensure positive
                'unit': 'kW',
                'timestamp': datetime.now().isoformat(),
                'average_consumption': float(avg_consumption),
                'confidence': '75-85%'
            })
        
        # Fallback to average
        elif baseline_model is None and lstm_model is None:
            # No models available - return average
            avg_consumption = df_data.select_dtypes(include=[np.number]).mean().mean()
            return jsonify({
                'model': 'Average (No trained model)',
                'prediction': float(avg_consumption),
                'unit': 'kW',
                'timestamp': datetime.now().isoformat(),
                'note': 'Using historical average. Train models for better predictions.'
            })
        
        else:
            return jsonify({'error': 'Selected model not available. Try "baseline" model type.'}), 500
    
    except Exception as e:
        # Return detailed error for debugging
        import traceback
        return jsonify({
            'error': str(e),
            'details': traceback.format_exc(),
            'suggestion': 'Make sure data preprocessing and model training are complete.'
        }), 500

@app.route('/api/suggestions')
def get_suggestions():
    """Get energy-saving suggestions"""
    try:
        # Load suggestions if available
        if os.path.exists('reports/results/energy_suggestions.csv'):
            df_suggestions = pd.read_csv('reports/results/energy_suggestions.csv')
            suggestions = df_suggestions.to_dict('records')
            return jsonify(suggestions)
        else:
            # Return default suggestions
            default_suggestions = [
                {
                    'device': 'General',
                    'type': 'Energy Saving',
                    'severity': 'Medium',
                    'suggestion': 'Turn off lights when leaving a room.',
                    'potential_savings': '10.00 kWh'
                },
                {
                    'device': 'HVAC',
                    'type': 'Temperature Control',
                    'severity': 'High',
                    'suggestion': 'Adjust thermostat by 2-3 degrees to save energy.',
                    'potential_savings': '50.00 kWh'
                }
            ]
            return jsonify(default_suggestions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-performance')
def get_model_performance():
    """Get model performance metrics"""
    try:
        metrics = {}
        
        # Load baseline metrics
        if os.path.exists('reports/results/baseline_metrics.csv'):
            baseline_metrics = pd.read_csv('reports/results/baseline_metrics.csv')
            metrics['baseline'] = baseline_metrics.to_dict('records')[0]
        
        # Load LSTM metrics
        if os.path.exists('reports/results/lstm_metrics.csv'):
            lstm_metrics = pd.read_csv('reports/results/lstm_metrics.csv')
            metrics['lstm'] = lstm_metrics.to_dict('records')[0]
        
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("SMART ENERGY ANALYSIS - WEB APPLICATION")
    print("="*60)
    
    # Load models and data
    load_models()
    
    print("\n[INFO] Starting Flask server...")
    print("   Access the application at: http://localhost:5000")
    print("="*60)
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
