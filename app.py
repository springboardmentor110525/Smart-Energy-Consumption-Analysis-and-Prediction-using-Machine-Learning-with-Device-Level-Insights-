import pickle
import os

from flask import Flask, render_template, request, jsonify

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


if __name__ == "__main__":
    app.run(debug=True)
