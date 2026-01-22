
import os
import numpy as np
from flask import jsonify
from flask import Flask, render_template, request, jsonify
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "linear_regression_24hr_model.pkl")

lr_model = joblib.load(MODEL_PATH)

print("✅ 24-hour Linear Regression model loaded successfully")


lr_model = joblib.load(MODEL_PATH)

model = joblib.load(MODEL_PATH)
print("✅ 24-hour Linear Regression model loaded successfully")

# Model evaluation metrics (from test data)
MODEL_METRICS = {
    "mae": 46.51,
    "rmse": 87.42,
    "r2": 0.96
}


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    devices = {
        "Air Conditioner": float(data["ac"]),
        "Fan": float(data["fan"]),
        "Fridge": float(data["fridge"]),
        "TV": float(data["tv"]),
        "Washing Machine": float(data["wm"])
    }

    max_device = max(devices, key=devices.get)
    min_device = min(devices, key=devices.get)

    tips = []
    if devices[max_device] > 500:
        tips.append(f"⚠️ {max_device} is consuming high power. Reduce usage or use energy-efficient mode.")
    if devices["Air Conditioner"] > 400:
        tips.append("🌬️ Set AC temperature to 24–26°C for energy savings.")
    if devices["Fridge"] > 300:
        tips.append("🧊 Avoid frequent fridge door opening.")
    tips.append("💡 Switch off devices when not in use.")

    return jsonify({
        "devices": devices,
        "max_device": max_device,
        "min_device": min_device,
        "tips": tips
    })

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    values = data.get("values", [])

    if len(values) != 24:
        return jsonify({"error": "Exactly 24 values required"}), 400

    X = np.array(values).reshape(1, -1)
    prediction = lr_model.predict(X)[0]

    return jsonify({
        "predicted_units": round(float(prediction), 3)
    })










if __name__ == "__main__":
    app.run(debug=True)
