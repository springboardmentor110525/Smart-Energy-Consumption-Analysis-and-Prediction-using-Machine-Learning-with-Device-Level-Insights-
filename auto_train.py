"""
Auto-Training Module for Smart Energy Consumption Analysis
Handles dataset upload validation, preprocessing, and model retraining.
"""
import os
import sys
import pickle
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global training status
training_status = {
    'is_training': False,
    'progress': 0,
    'current_step': 'Idle',
    'error': None,
    'last_trained': None,
    'history': []
}

# Required columns for validation
REQUIRED_COLUMNS = ['use [kW]']
TIME_COLUMNS = ['time', 'datetime', 'date', 'timestamp']

# Device columns (optional)
DEVICE_COLUMNS = [
    'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
    'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
    'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
    'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
    'Microwave [kW]', 'Living room [kW]', 'Solar [kW]'
]


def get_training_status():
    """Return current training status."""
    return training_status.copy()


def validate_csv(filepath):
    """
    Validate that uploaded CSV has required columns.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (is_valid, message, dataframe or None)
    """
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Check for time column
        time_col = None
        for col in TIME_COLUMNS:
            if col in df.columns:
                time_col = col
                break
            # Case insensitive check
            for c in df.columns:
                if c.lower() == col.lower():
                    time_col = c
                    break
            if time_col:
                break
        
        if not time_col:
            return False, f"Missing time column. Expected one of: {TIME_COLUMNS}", None
        
        # Check for use [kW] column
        use_col = None
        for col in df.columns:
            if 'use' in col.lower() and 'kw' in col.lower():
                use_col = col
                break
        
        if not use_col:
            return False, "Missing 'use [kW]' column for total energy consumption", None
        
        # Check for minimum rows
        if len(df) < 24:
            return False, f"Dataset too small. Need at least 24 rows, got {len(df)}", None
        
        # Parse time column
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        except Exception as e:
            return False, f"Could not parse time column '{time_col}': {str(e)}", None
        
        return True, f"Valid CSV with {len(df)} rows", df
        
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}", None


def preprocess_data(df):
    """
    Preprocess the data for training.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Tuple of (train_df, test_df) with engineered features
    """
    # Find the use column
    use_col = None
    for col in df.columns:
        if 'use' in col.lower() and 'kw' in col.lower():
            use_col = col
            break
    
    if use_col != 'use [kW]':
        df = df.rename(columns={use_col: 'use [kW]'})
    
    # Handle missing values
    df = df.dropna(subset=['use [kW]'])
    
    # Create time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Create hour categories
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Create lag features (if enough data)
    if len(df) > 24:
        df['use_lag_1'] = df['use [kW]'].shift(1)
        df['use_lag_24'] = df['use [kW]'].shift(24)
        df['rolling_mean_24'] = df['use [kW]'].rolling(window=24).mean()
    
    # Drop rows with NaN from lag features
    df = df.dropna()
    
    # Select feature columns
    feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 
                   'hour_sin', 'hour_cos']
    
    if 'use_lag_1' in df.columns:
        feature_cols.extend(['use_lag_1', 'use_lag_24', 'rolling_mean_24'])
    
    # Add device columns if present
    for col in DEVICE_COLUMNS:
        if col in df.columns:
            feature_cols.append(col)
    
    # Scale features
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Split into train/test (80/20)
    split_idx = int(len(df_scaled) * 0.8)
    train_df = df_scaled.iloc[:split_idx][feature_cols + ['use [kW]']]
    test_df = df_scaled.iloc[split_idx:][feature_cols + ['use [kW]']]
    
    # Save processed data
    df.to_csv('clean_energy_data.csv')
    train_df.to_csv('train_features.csv')
    test_df.to_csv('test_features.csv')
    
    # Save scaler for later use
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return train_df, test_df


def train_linear_regression(train_df, test_df):
    """
    Train Linear Regression model.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        
    Returns:
        Dictionary with metrics
    """
    target_col = 'use [kW]'
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/linear_regression.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save predictions for comparison
    os.makedirs('results', exist_ok=True)
    results = {
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': metrics,
        'index': test_df.index
    }
    with open('results/baseline_predictions.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return metrics


def train_lstm_model(train_df, test_df, time_steps=12, epochs=30, batch_size=8):
    """
    Train LSTM model.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        time_steps: Number of time steps for sequences
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with metrics
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        return {'error': 'TensorFlow not installed'}
    
    target_col = 'use [kW]'
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Create sequences
    def create_sequences(X, y, time_steps):
        X_seq, y_seq = [], []
        for i in range(time_steps, len(X)):
            X_seq.append(X[i - time_steps:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    # Adjust time_steps if needed
    if len(X_train) < time_steps + 10:
        time_steps = max(1, len(X_train) // 3)
    
    if time_steps < 2:
        # Fallback: use simple reshaping
        X_train_seq = X_train.reshape(-1, 1, X_train.shape[1])
        y_train_seq = y_train
        X_test_seq = X_test.reshape(-1, 1, X_test.shape[1])
        y_test_seq = y_test
    else:
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)
    
    if len(X_test_seq) == 0:
        X_test_seq = X_test.reshape(-1, 1, X_test.shape[1])
        y_test_seq = y_test
    
    # Build model
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    ]
    
    # Train
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    
    # Predictions
    y_pred = model.predict(X_test_seq, verbose=0).flatten()
    
    # Metrics
    metrics = {
        'mae': float(mean_absolute_error(y_test_seq, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test_seq, y_pred))),
        'r2': float(r2_score(y_test_seq, y_pred))
    }
    
    # Save model
    model.save('models/lstm_model.keras')
    
    # Save predictions
    results = {
        'y_test': y_test_seq,
        'y_pred': y_pred,
        'metrics': metrics,
        'history': history.history
    }
    with open('results/lstm_predictions.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return metrics


def run_full_training(filepath):
    """
    Run the complete training pipeline in a thread.
    
    Args:
        filepath: Path to uploaded CSV file
    """
    global training_status
    
    training_status['is_training'] = True
    training_status['progress'] = 0
    training_status['error'] = None
    
    try:
        # Step 1: Validate CSV (10%)
        training_status['current_step'] = 'Validating CSV...'
        training_status['progress'] = 10
        
        is_valid, message, df = validate_csv(filepath)
        if not is_valid:
            training_status['error'] = message
            training_status['is_training'] = False
            return
        
        # Step 2: Preprocess data (30%)
        training_status['current_step'] = 'Preprocessing data...'
        training_status['progress'] = 30
        
        train_df, test_df = preprocess_data(df)
        
        # Step 3: Train Linear Regression (50%)
        training_status['current_step'] = 'Training Linear Regression model...'
        training_status['progress'] = 50
        
        lr_metrics = train_linear_regression(train_df, test_df)
        
        # Step 4: Train LSTM (90%)
        training_status['current_step'] = 'Training LSTM model...'
        training_status['progress'] = 70
        
        lstm_metrics = train_lstm_model(train_df, test_df)
        
        # Complete (100%)
        training_status['progress'] = 100
        training_status['current_step'] = 'Training complete!'
        training_status['last_trained'] = datetime.now().isoformat()
        
        # Add to history
        training_status['history'].insert(0, {
            'timestamp': training_status['last_trained'],
            'dataset': os.path.basename(filepath),
            'rows': len(df),
            'lr_r2': round(lr_metrics.get('r2', 0), 4),
            'lstm_r2': round(lstm_metrics.get('r2', 0), 4) if 'error' not in lstm_metrics else 'N/A'
        })
        
        # Keep only last 10 training runs
        training_status['history'] = training_status['history'][:10]
        
    except Exception as e:
        training_status['error'] = str(e)
    finally:
        training_status['is_training'] = False


def start_training_async(filepath):
    """Start training in a background thread."""
    if training_status['is_training']:
        return False, "Training already in progress"
    
    thread = threading.Thread(target=run_full_training, args=(filepath,))
    thread.daemon = True
    thread.start()
    
    return True, "Training started"


def get_dataset_info():
    """Get information about the current dataset."""
    info = {
        'filename': None,
        'rows': 0,
        'date_range': {'start': None, 'end': None},
        'last_trained': training_status.get('last_trained'),
        'columns': []
    }
    
    try:
        if os.path.exists('clean_energy_data.csv'):
            df = pd.read_csv('clean_energy_data.csv', index_col=0, parse_dates=True, nrows=5)
            full_df = pd.read_csv('clean_energy_data.csv', index_col=0, parse_dates=True)
            
            info['filename'] = 'clean_energy_data.csv'
            info['rows'] = len(full_df)
            info['date_range'] = {
                'start': str(full_df.index.min()),
                'end': str(full_df.index.max())
            }
            info['columns'] = list(full_df.columns)
            
            # Get file modification time
            mtime = os.path.getmtime('clean_energy_data.csv')
            info['last_modified'] = datetime.fromtimestamp(mtime).isoformat()
    except Exception as e:
        info['error'] = str(e)
    
    return info


if __name__ == '__main__':
    # Test validation
    print("Testing auto_train module...")
    print(f"Current dataset info: {get_dataset_info()}")
