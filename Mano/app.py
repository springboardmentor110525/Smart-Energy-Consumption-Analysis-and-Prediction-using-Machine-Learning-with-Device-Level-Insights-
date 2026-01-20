from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\91936\OneDrive\Desktop\Mano"
MODEL_PATH = os.path.join(BASE_DIR, "energy_model.h5")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "model_artifacts.pkl")

print(f"Looking for model at: {MODEL_PATH}")

# ==========================================
# LOAD AI ENGINE
# ==========================================
try:
    # compile=False fixes the 'mse' error
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    with open(ARTIFACTS_PATH, 'rb') as f:
        artifacts = pickle.load(f)

    scaler_X = artifacts['scaler_X']
    scaler_y = artifacts['scaler_y']
    le_room = artifacts['le_room']
    le_device = artifacts['le_device']
    le_day_type = artifacts['le_day_type']
    
    print("✅ AI Engine Loaded & Ready!")

except Exception as e:
    print(f"\n❌ ERROR: {e}\n")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Extract Base Inputs
        home_id = 1 
        room = data['room']
        device = data['device']
        status = int(data['status'])
        power_w = float(data['power_w'])
        # We ignore user 'hour' for the long-term prediction, 
        # but keep it for the "Current Instant" prediction.
        current_hour = int(data['hour']) 
        day_of_week = int(data['day_of_week'])
        day_type = data['day_type']

        # 2. Encode Categories
        if room not in le_room.classes_ or device not in le_device.classes_:
            return jsonify({'error': "Unknown Room or Device"})
            
        room_enc = le_room.transform([room])[0]
        device_enc = le_device.transform([device])[0]
        day_type_enc = le_day_type.transform([day_type])[0]

        # ==========================================================
        # 🧪 PREDICTION 1: INSTANTANEOUS (Current Hour)
        # ==========================================================
        # Input: [home_id, room, device, status, power, hour, day_week, day_type]
        features_instant = np.array([[
            home_id, room_enc, device_enc, status, power_w, current_hour, day_of_week, day_type_enc
        ]])
        
        # Scale & Predict
        feat_inst_scaled = scaler_X.transform(features_instant)
        feat_inst_reshaped = feat_inst_scaled.reshape((1, 1, feat_inst_scaled.shape[1]))
        pred_instant_scaled = model.predict(feat_inst_reshaped)
        pred_instant_kwh = scaler_y.inverse_transform(pred_instant_scaled)[0][0]

        # ==========================================================
        # 🔮 PREDICTION 2: NEXT 24 HOURS (Batch Simulation)
        # ==========================================================
        # We create 24 rows of data, one for each hour of the day (0-23)
        # to see how the device behaves throughout a full day.
        
        # Create 24 copies of the input
        batch_features = np.repeat(features_instant, 24, axis=0)
        
        # Overwrite the 'hour' column (index 5) with 0, 1, 2... 23
        batch_features[:, 5] = np.arange(24)
        
        # Scale & Predict Batch
        batch_scaled = scaler_X.transform(batch_features)
        batch_reshaped = batch_scaled.reshape((24, 1, batch_scaled.shape[1]))
        
        batch_predictions_scaled = model.predict(batch_reshaped)
        batch_predictions_kwh = scaler_y.inverse_transform(batch_predictions_scaled)
        
        # Sum up all hourly predictions to get Total Daily Consumption
        # Ensure no negative values (physics check)
        daily_total = np.sum(np.maximum(batch_predictions_kwh, 0))

        # ==========================================================
        # 📅 PREDICTION 3: WEEK & MONTH EXTRAPOLATION
        # ==========================================================
        weekly_total = daily_total * 7
        monthly_total = daily_total * 30

        return jsonify({
            'instant_kwh': f"{float(pred_instant_kwh):.4f}",
            'day_kwh': f"{float(daily_total):.2f}",
            'week_kwh': f"{float(weekly_total):.2f}",
            'month_kwh': f"{float(monthly_total):.2f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)