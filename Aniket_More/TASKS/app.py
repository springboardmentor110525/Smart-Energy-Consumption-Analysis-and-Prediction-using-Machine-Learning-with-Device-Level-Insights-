from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --------------------------------------------------
# Load model & preprocessing objects
# --------------------------------------------------
model = load_model("models/best_lstm_model.keras")
scaler = joblib.load("models/lstm_scaler.pkl")
feature_cols = joblib.load("models/lstm_feature_cols.pkl")
timesteps = joblib.load("models/lstm_timesteps.pkl")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
data = pd.read_csv(
    r"D:\Infosys_Virtual_Intership\Smart_Energy_Consumption_Analysis_and_Prediction_using_ML\TASKS\dataset\HomeC_augmented.csv"
)

# Ensure all feature columns exist
for col in feature_cols:
    if col not in data.columns:
        data[col] = 0.0

data[feature_cols] = data[feature_cols].astype(float)

# --------------------------------------------------
# Device list
# --------------------------------------------------
TIME_WEATHER = [
    'hour', 'day', 'weekofyear', 'month',
    'temperature', 'humidity', 'pressure',
    'windSpeed', 'cloudCover'
]

devices = [c for c in feature_cols if c not in TIME_WEATHER]

# --------------------------------------------------
# Smart tips
# --------------------------------------------------
SMART_TIPS = {
    "Dishwasher": "💡 Run only full loads using eco mode.",
    "Air conditioning [kW]": "💡 Set AC to 24°C for energy saving.",
    "Fridge": "💡 Keep fridge doors closed and clean coils.",
    "Laundry [kW]": "💡 Wash clothes in cold water.",
    "Home office": "💡 Turn off devices when not in use.",
    "Microwave": "💡 Avoid reheating food multiple times.",
    "Outdoor lights [kW]": "❄️ Outdoor lights is turned OFF."
}

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predictions.html", devices=devices)

@app.route("/home-content")
def home_content():
    return render_template("home_content.html")

@app.route("/devices")
def devices_page():
    return render_template("devices.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/reports")
def reports_page():
    return render_template("reports.html")


@app.route("/suggestion")
def suggestion_page():
    return render_template("suggestion.html")


@app.route("/settings")
def settings_page():
    return render_template("settings.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.get_json()

        current_energy = float(req.get("current_energy", 0))
        device = req.get("device")
        time_feature = req.get("time_feature", "hourly").lower()

        # -----------------------------
        # Validation
        # -----------------------------
        if device not in devices:
            return jsonify({"error": "Invalid device selected"})

        # 🔴 DEVICE OFF LOGIC (IMPORTANT)
        if current_energy == 0:
            return jsonify({
                "prediction_kWh": 0,
                "trend": [],
                "smart_tip": f"❄️ {device} is turned OFF."
            })

        # --------------------------------------------------
        # 1️⃣ HARD VALIDATION (MOST IMPORTANT)
        # --------------------------------------------------
        current_energy = max(current_energy, 0)

        # AC realistic hourly range
        if device == "Air conditioning [kW]":
            current_energy = max(0.8, min(current_energy, 4.5))

        # Dishwasher realistic range
        if device == "Dishwasher":
            current_energy = max(0.5, min(current_energy, 6))

        # Outdoor lights OFF case
        if device == "Outdoor lights [kW]" and current_energy == 0:
            return jsonify({
                "prediction_kWh": 0,
                "trend": [],
                "smart_tip": SMART_TIPS[device]
            })

        # --------------------------------------------------
        # 2️⃣ Prepare LSTM input with PATTERN INJECTION
        # --------------------------------------------------
        X = data[feature_cols].copy()
        idx = X.columns.get_loc(device)

        # Inject gradual pattern instead of flat value
        pattern = np.linspace(
            current_energy * 0.85,
            current_energy,
            timesteps
        )

        X.iloc[-timesteps:, idx] = pattern

        X_seq = X.tail(timesteps).values
        X_seq = scaler.transform(X_seq)
        X_seq = X_seq.reshape(1, timesteps, len(feature_cols))

        # --------------------------------------------------
        # 3️⃣ Hourly prediction
        # --------------------------------------------------
        pred_hourly = float(model.predict(X_seq, verbose=0)[0][0])
        pred_hourly = max(pred_hourly, 0)

        # --------------------------------------------------
        # 4️⃣ Convert to selected time feature
        # --------------------------------------------------
        if time_feature == "hourly":
            final_pred = pred_hourly
            steps = 24
            label = "Hour"

        elif time_feature == "daily":
            final_pred = pred_hourly * 24
            steps = 7
            label = "Day"

        elif time_feature == "weekly":
            final_pred = pred_hourly * 24 * 7
            steps = 4
            label = "Week"

        elif time_feature == "monthly":
            final_pred = pred_hourly * 24 * 30
            steps = 12
            label = "Month"

        else:
            final_pred = pred_hourly
            steps = 24
            label = "Hour"

        final_pred = round(final_pred, 2)

        # --------------------------------------------------
        # 5️⃣ Trend generation
        # --------------------------------------------------
        trend = []
        temp_seq = X_seq.copy()

        for i in range(steps):
            y = float(model.predict(temp_seq, verbose=0)[0][0])
            y = max(y, 0)

            trend.append({
                "time": f"{label} {i + 1}",
                "value": round(y, 2)
            })

            temp_seq = np.roll(temp_seq, -1, axis=1)
            temp_seq[0, -1, idx] = y

        # --------------------------------------------------
        # Response
        # --------------------------------------------------
        return jsonify({
            "prediction_kWh": final_pred,
            "trend": trend,
            "smart_tip": SMART_TIPS.get(device, "💡 Save energy wherever possible.")
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed. Please try again."})

# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
