"""
Module 5 & 6: Advanced Forecasting - LSTM Model
Trains an LSTM model on the processed energy data for time series forecasting.
"""
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.utils import load_data, prepare_data, evaluate_model, create_lstm_sequences

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    print("TensorFlow/Keras not installed. Please install with: pip install tensorflow")
    sys.exit(1)


def build_lstm_model(input_shape, units=64, dropout=0.2):
    """
    Build an LSTM model for energy consumption forecasting.
    
    Args:
        input_shape: Tuple of (time_steps, features)
        units: Number of LSTM units
        dropout: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_lstm(time_steps=24, epochs=100, batch_size=16):
    """Train and evaluate the LSTM model."""
    print("Loading data...")
    train_df, test_df = load_data()
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    
    # Prepare features and target
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(train_df, test_df)
    
    print(f"\nFeatures used: {len(feature_cols)}")
    
    # Create sequences for LSTM
    print(f"\nCreating sequences with {time_steps} time steps...")
    X_train_seq, y_train_seq = create_lstm_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_lstm_sequences(X_test, y_test, time_steps)
    
    print(f"X_train_seq shape: {X_train_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}")
    
    if len(X_test_seq) == 0:
        print("\nWarning: Not enough test data for the specified time steps.")
        print("Reducing time_steps to work with available data...")
        time_steps = max(1, len(X_test) - 1)
        X_train_seq, y_train_seq = create_lstm_sequences(X_train, y_train, time_steps)
        X_test_seq, y_test_seq = create_lstm_sequences(X_test, y_test, time_steps)
        print(f"New X_train_seq shape: {X_train_seq.shape}")
        print(f"New X_test_seq shape: {X_test_seq.shape}")
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print("\nError: Not enough data for LSTM training. Using simpler approach...")
        # Fallback: use last few rows as simple sequences
        min_samples = 2
        time_steps = 1
        X_train_seq = X_train.reshape(-1, 1, X_train.shape[1])
        y_train_seq = y_train
        X_test_seq = X_test.reshape(-1, 1, X_test.shape[1])
        y_test_seq = y_test
        print(f"Fallback X_train_seq shape: {X_train_seq.shape}")
        print(f"Fallback X_test_seq shape: {X_test_seq.shape}")
    
    # Build model
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    print(f"\nBuilding LSTM model with input shape: {input_shape}")
    model = build_lstm_model(input_shape)
    model.summary()
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/lstm_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Make predictions
    y_train_pred = model.predict(X_train_seq).flatten()
    y_test_pred = model.predict(X_test_seq).flatten()
    
    # Evaluate
    print("\n--- Training Set ---")
    train_metrics = evaluate_model(y_train_seq, y_train_pred, "LSTM (Train)")
    
    print("\n--- Test Set ---")
    test_metrics = evaluate_model(y_test_seq, y_test_pred, "LSTM (Test)")
    
    # Save predictions for comparison
    os.makedirs('results', exist_ok=True)
    results = {
        'y_test': y_test_seq,
        'y_pred': y_test_pred,
        'metrics': test_metrics,
        'history': history.history
    }
    with open('results/lstm_predictions.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    
    # MAE plot
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training History - MAE')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('results/lstm_training_history.png', dpi=150)
    plt.close()
    print("\nTraining history saved to results/lstm_training_history.png")
    
    # Predictions plot
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(y_test_seq)), y_test_seq, label='Actual', alpha=0.7)
    plt.plot(range(len(y_test_seq)), y_test_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Energy Consumption (Normalized)')
    plt.title('LSTM: Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/lstm_predictions.png', dpi=150)
    plt.close()
    print("Predictions saved to results/lstm_predictions.png")
    
    print("\nLSTM model saved to models/lstm_model.keras")
    
    return model, test_metrics


if __name__ == "__main__":
    train_lstm(time_steps=12, epochs=50, batch_size=8)
