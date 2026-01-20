"""
Configuration settings for Smart Energy Analysis System
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
DATASET_PATH = BASE_DIR / 'HomeC_augmented.csv'

# Model paths
MODEL_DIR = BASE_DIR / 'models'
BASELINE_MODEL_PATH = MODEL_DIR / 'baseline_lr.pkl'
LSTM_MODEL_PATH = MODEL_DIR / 'best_lstm_model.h5'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'

# Report paths
REPORT_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORT_DIR / 'figures'
RESULTS_DIR = REPORT_DIR / 'results'

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODEL_DIR, REPORT_DIR, FIGURES_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model hyperparameters
LSTM_CONFIG = {
    'sequence_length': 24,  # Use last 24 hours to predict next hour
    'units': 128,
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'validation_split': 0.2
}

BASELINE_CONFIG = {
    'test_size': 0.2,
    'random_state': 42
}

# Feature engineering
TIME_FEATURES = ['hour', 'day', 'month', 'weekday', 'weekofyear']
DEVICE_COLUMNS = [
    'Dishwasher', 'Home office', 'Fridge', 'Wine cellar', 
    'Garage door', 'Barn', 'Well', 'Microwave', 'Living room',
    'Furnace', 'Kitchen', 'Car charger [kW]', 'Water heater [kW]',
    'Air conditioning [kW]', 'Home Theater [kW]', 'Outdoor lights [kW]',
    'microwave [kW]', 'Laundry [kW]', 'Pool Pump [kW]'
]

WEATHER_FEATURES = [
    'temperature', 'humidity', 'visibility', 'apparentTemperature',
    'pressure', 'windSpeed', 'cloudCover', 'windBearing',
    'precipIntensity', 'dewPoint', 'precipProbability'
]

# Target variable
TARGET_COLUMN = 'use_HO'  # Total home office usage (can be changed)

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 6)
DPI = 100

# Flask app settings
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'SECRET_KEY': 'smart-energy-analysis-secret-key-2026'
}

# Energy saving thresholds
ENERGY_THRESHOLDS = {
    'high_usage': 0.5,  # kW
    'medium_usage': 0.2,  # kW
    'low_usage': 0.1  # kW
}

# Suggestions database
ENERGY_TIPS = {
    'Dishwasher': 'Run dishwasher only when full. Use eco mode if available.',
    'Fridge': 'Keep refrigerator at optimal temperature (37-40°F). Clean coils regularly.',
    'Microwave': 'Use microwave instead of oven for small meals to save energy.',
    'Air conditioning [kW]': 'Set AC to 24-26°C. Use programmable thermostat.',
    'Water heater [kW]': 'Lower water heater temperature to 120°F. Insulate tank.',
    'Furnace': 'Regular maintenance and filter changes improve efficiency.',
    'Pool Pump [kW]': 'Run pool pump during off-peak hours. Consider variable speed pump.',
    'Laundry [kW]': 'Wash clothes in cold water. Run full loads only.',
    'Car charger [kW]': 'Charge during off-peak hours for lower rates.',
    'Outdoor lights [kW]': 'Use LED bulbs and motion sensors. Install timers.',
    'Home Theater [kW]': 'Enable power-saving mode. Unplug when not in use.',
    'Living room': 'Use LED bulbs. Turn off lights when leaving room.',
    'Home office': 'Enable sleep mode on computers. Unplug chargers when not in use.',
    'Kitchen': 'Use energy-efficient appliances. Avoid opening oven door frequently.'
}

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# API settings
API_VERSION = 'v1'
MAX_PREDICTION_DAYS = 30

print("✓ Configuration loaded successfully")
