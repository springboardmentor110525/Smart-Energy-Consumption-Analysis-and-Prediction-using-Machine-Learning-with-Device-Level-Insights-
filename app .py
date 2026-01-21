from flask import Flask, render_template, request, jsonify, Response
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import io
import traceback
import numpy as np

app = Flask(__name__, template_folder=".")

# --- Load initial dataset ---
df = pd.read_csv("processed_energy_data.csv", index_col=0, parse_dates=True)
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

# --- Load model artifacts ---
model_type = None
pipeline = None
tuned_lstm = None
scaler_X = None
timesteps = None
feature_cols = []
default_values = {}

try:
    model_type = joblib.load("model_type.pkl")
except:
    model_type = None

if model_type == "LR":
    pipeline = joblib.load("energy_pipeline.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    default_values = joblib.load("default_values.pkl")
elif model_type == "LSTM":
    from tensorflow.keras.models import load_model
    tuned_lstm = load_model("tuned_lstm_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    timesteps = joblib.load("timesteps.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    default_values = joblib.load("default_values.pkl")

# --- Smart tips dictionary ---
smart_tips = {
    "Fridge": "Check door seals and avoid frequent opening to save energy.",
    "Microwave": "Avoid running empty and use appropriate containers.",
    "Air Conditioner": "Clean filters regularly and set thermostat efficiently.",
    "Dishwasher": "Run full loads and use eco mode.",
    "Laundry": "Wash with cold water and dry clothes naturally."
}

# --- Helper functions ---
def predict_energy(input_dict):
    if model_type is None:
        return 0.0

    row = {}
    for col in feature_cols:
        if col.startswith("device_"):
            row[col] = 1 if col == f"device_{input_dict.get('device','')}" else 0
        elif col.startswith("time_feature_"):
            row[col] = 1 if col == f"time_feature_{input_dict.get('time_feature','')}" else 0
        else:
            row[col] = input_dict.get(col, default_values.get(col, 0))

    input_df = pd.DataFrame([row])[feature_cols]

    if model_type == "LR":
        pred = pipeline.predict(input_df)
        return float(pred[0])
    elif model_type == "LSTM":
        input_scaled = scaler_X.transform(input_df)
        input_seq = np.repeat(input_scaled, timesteps, axis=0).reshape(1, timesteps, -1)
        pred = tuned_lstm.predict(input_seq).flatten()[0]
        return float(pred)

def get_trend(device, time_feature):
    if time_feature == "hourly":
        df_resampled = df.resample("h").sum()
    elif time_feature == "daily":
        df_resampled = df.resample("D").sum()
    elif time_feature == "weekly":
        df_resampled = df.resample("W").sum()
    elif time_feature == "monthly":
        df_resampled = df.resample("M").sum()
    else:
        df_resampled = df.resample("D").sum()

    trend = []
    if device in df_resampled.columns:
        for idx, val in df_resampled[device].tail(20).items():
            trend.append({"time": str(idx), "value": round(val, 3)})
    else:
        for idx, val in df_resampled["total_energy"].tail(20).items():
            trend.append({"time": str(idx), "value": round(val, 3)})
    return trend

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/plot/overview")
def plot_overview():
    df_hourly = df.resample("h").sum()
    df_daily = df.resample("D").sum()
    df_weekly = df.resample("W").sum()
    df_monthly = df.resample("M").sum()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Dashboard components for hourly/daily/weekly/monthly consumption",
                 fontsize=18, fontweight="bold")

    df_hourly["total_energy"].plot(ax=axes[0,0], color="blue", linewidth=2, label="Hourly")
    axes[0,0].set_title("Hourly Energy Consumption"); axes[0,0].legend(); axes[0,0].grid(True)

    df_daily["total_energy"].plot(ax=axes[0,1], color="green", linewidth=2, label="Daily")
    axes[0,1].set_title("Daily Energy Consumption"); axes[0,1].legend(); axes[0,1].grid(True)

    df_weekly["total_energy"].plot(ax=axes[1,0], color="red", linewidth=2, label="Weekly")
    axes[1,0].set_title("Weekly Energy Consumption"); axes[1,0].legend(); axes[1,0].grid(True)

    df_monthly["total_energy"].plot(kind="bar", ax=axes[1,1], color="purple", width=0.8, label="Monthly")
    axes[1,1].set_title("Monthly Energy Consumption"); axes[1,1].legend(); axes[1,1].grid(True)
    for tick in axes[1,1].get_xticklabels(): tick.set_rotation(45)

    plt.tight_layout(rect=[0,0,1,0.96])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

@app.route("/plot/devices")
def plot_devices():
    """Pie chart of energy consumption by your feature columns."""
    df_resampled = df.resample("D").sum()
    device_cols = [c for c in df_resampled.columns if c != "total_energy"]

    if not device_cols:
        device_totals = pd.Series({"total_energy": df_resampled["total_energy"].sum()})
    else:
        device_totals = df_resampled[device_cols].sum()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(device_totals, labels=device_totals.index, autopct="%1.1f%%",
           startangle=140, colors=plt.cm.tab20.colors)
    ax.set_title("Total Energy Consumption by Feature", fontsize=16)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        current_energy = float(data.get("current_energy", 0))
        device = data.get("device", "Fridge")
        time_feature = data.get("time_feature", "daily")

        now = pd.Timestamp.now()

        input_dict = {
            "hour": now.hour,
            "day": now.day,
            "month": now.month,
            "is_weekend": int(now.dayofweek >= 5),
            "lag_1": df["total_energy"].iloc[-1],
            "lag_24": df["total_energy"].iloc[-24] if len(df) >= 24 else current_energy,
            "ma_3": df["total_energy"].tail(3).mean(),
            "ma_24": df["total_energy"].tail(24).mean(),
            "current_energy": current_energy,
            "device": device,
            "time_feature": time_feature
        }

        prediction = predict_energy(input_dict)
        tip = smart_tips.get(device, "Use energy wisely and monitor device usage.")
        trend = get_trend(device, time_feature)

        return jsonify({
            "prediction_kWh": round(prediction, 2),
            "smart_tip": tip,
            "trend": trend,
            "model_type": model_type
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
