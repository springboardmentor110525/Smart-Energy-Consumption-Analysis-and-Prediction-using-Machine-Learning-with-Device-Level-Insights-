# test_lstm_predictions.py

import numpy as np
from tensorflow.keras.models import load_model

#Load the saved LSTM model
model = load_model("best_lstm_model.keras")
print("LSTM model loaded successfully")

#Define a sample input
# Suppose your model expects 'timesteps' x 'features' shape
# Example: timesteps = 10, features = 8
timesteps = 10
features = 9

# Create a sample input (replace with actual feature values)
# Shape must be (1, timesteps, features) for a single prediction
# Example: 10 time steps of real data
sample_input = np.array([[
    [12, 3, 2, 1, 30.5, 60, 1012, 5, 0.2],  # timestep 1
    [13, 3, 2, 1, 31.0, 61, 1013, 6, 0.1],  # timestep 2
    [21, 3, 2, 1, 29.5, 55, 1011, 4, 0.3]   # timestep 10
]]).astype(np.float32)


#Make prediction
predicted_value = model.predict(sample_input)
print("Predicted value:", predicted_value)
