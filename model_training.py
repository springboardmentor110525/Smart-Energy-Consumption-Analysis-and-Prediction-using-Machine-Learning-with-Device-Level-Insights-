import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)   # moves from app/ to project root


print("🔹 Loading dataset...")

# Load dataset
df = pd.read_csv("D:\DomEnergyProject\Smart_Energy_Consumption_Analysis\data\energy_data.csv")


# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by time
df = df.sort_values('timestamp')

# Target column
TARGET = 'power_consumption'

# Create 24 lag features
for i in range(1, 25):
    df[f'lag_{i}'] = df[TARGET].shift(i)

# Drop rows with NaN
df.dropna(inplace=True)

# Features and target
X = df[[f'lag_{i}' for i in range(1, 25)]]
y = df[TARGET]

print("🔹 Feature shape:", X.shape)

# Time-aware split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("✅ Model Evaluation Metrics")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# Save model inside app folder
# Save model safely inside app folder
MODEL_DIR = os.path.join(BASE_DIR, "app")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression_24hr_model.pkl")
joblib.dump(model, MODEL_PATH)

print(f"🎯 Model saved at: {MODEL_PATH}")


print("🎯 Model saved as app/linear_regression_24hr_model.pkl")
