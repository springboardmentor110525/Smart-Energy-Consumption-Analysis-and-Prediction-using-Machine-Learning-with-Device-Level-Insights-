from flask import Flask, render_template, request, jsonify
from utils.predictor import predict_energy

app = Flask(__name__)

# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- FORECAST PAGE (FIXES 404) ----------------
@app.route("/forecast")
def forecast():
    return render_template("forecast.html")


# ---------------- PREDICTION API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        energy = float(data["energy"])
        device = data["device"]
        horizon = data["horizon"]

        dates, preds, tip = predict_energy(energy, device, horizon)

        return jsonify({
            "dates": dates,
            "predictions": preds,
            "tip": tip
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500



# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)
