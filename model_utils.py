import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Build model (same as training)
model = Sequential([
    Input(shape=(24, 1)),
    LSTM(32, return_sequences=True),
    LSTM(16),
    Dense(1)
])

# Load weights
model.load_weights("tuned_lstm_energy_model_weights.h5")

# Load scaler
y_scaler = joblib.load("energy_scaler.pkl")

def prepare_sequence(current_energy):
    """
    Convert single user input into 24-hour sequence
    """
    return np.array([current_energy] * 24)

def predict_energy_lstm(current_energy):
    seq_24 = prepare_sequence(current_energy)
    seq_24 = seq_24.reshape(1, 24, 1)

    pred_norm = model.predict(seq_24, verbose=0)
    pred_real = y_scaler.inverse_transform(pred_norm)

    return float(pred_real[0][0])
