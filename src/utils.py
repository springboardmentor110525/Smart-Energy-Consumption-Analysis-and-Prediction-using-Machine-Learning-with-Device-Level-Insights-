"""
Utility functions for the Smart Energy Consumption Analysis project.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data():
    """Load the training and testing feature datasets."""
    train_df = pd.read_csv('train_features.csv', index_col=0, parse_dates=True)
    test_df = pd.read_csv('test_features.csv', index_col=0, parse_dates=True)
    return train_df, test_df


def prepare_data(train_df, test_df, target_col='use [kW]'):
    """
    Prepare features (X) and target (y) for training and testing.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        target_col: Name of the target column
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculate and print evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model for display
    
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n=== {model_name} Evaluation ===")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2:   {r2:.6f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def create_lstm_sequences(X, y, time_steps=24):
    """
    Create sequences for LSTM input.
    
    Args:
        X: Feature array
        y: Target array
        time_steps: Number of time steps for each sequence
    
    Returns:
        X_seq, y_seq: 3D array for LSTM input, corresponding targets
    """
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X)):
        X_seq.append(X[i - time_steps:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)
