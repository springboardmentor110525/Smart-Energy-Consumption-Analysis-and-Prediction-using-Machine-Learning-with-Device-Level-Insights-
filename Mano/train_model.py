# train_model.py
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load Dataset (Make sure your CSV is in the same folder)
file_path = r"C:\Users\91936\Downloads\smart_home_6_months_dataset.csv" 
df = pd.read_csv(file_path)

# 2. Preprocessing
# We drop timestamp for the specific prediction logic, but keep other features
if 'timestamp' in df.columns:
    df = df.drop(columns=['timestamp'])

# Encode Categorical Variables (Room, Device, Day_Type)
le_room = LabelEncoder()
df['room'] = le_room.fit_transform(df['room'])

le_device = LabelEncoder()
df['device'] = le_device.fit_transform(df['device'])

le_day_type = LabelEncoder()
df['day_type'] = le_day_type.fit_transform(df['day_type'])

# Select Features and Target
# Inputs: home_id, room, device, status, power_w, hour, day_of_week, day_type
X = df.drop(columns=['energy_kwh']) 
y = df['energy_kwh']

# Scale Features (LSTMs work best with scaled data)
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape for LSTM [samples, time_steps, features]
# We treat each row as 1 time step for this specific implementation
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 3. Train Model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(1, X_scaled.shape[1])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print("Training Model... (This may take a moment)")
model.fit(X_reshaped, y_scaled, epochs=5, batch_size=32, verbose=1)

# 4. Save Artifacts for the Web App
model.save('energy_model.h5')

# We MUST save the scalers and encoders to apply the EXACT same math to user inputs
artifacts = {
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'le_room': le_room,
    'le_device': le_device,
    'le_day_type': le_day_type
}

with open('model_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Training complete! 'energy_model.h5' and 'model_artifacts.pkl' saved.")