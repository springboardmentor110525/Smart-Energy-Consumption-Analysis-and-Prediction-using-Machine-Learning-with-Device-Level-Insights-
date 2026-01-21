import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import random

model = load_model("models/lstm_energy_model.h5", compile=False)

# Realistic device scaling
DEVICE_MULTIPLIER = {
    "AC": 1.6,
    "Fridge": 1.2,
    "Washer": 1.3,
    "TV": 0.8,
    "Light": 0.4
}

# Daily fluctuation ranges (± %)
DEVICE_VARIANCE = {
    "AC": 0.15,
    "Fridge": 0.08,
    "Washer": 0.12,
    "TV": 0.10,
    "Light": 0.05
}

def predict_energy(last_energy, device, horizon):

    # ---------------- Horizon logic ----------------
    if horizon == "hour":
        internal_steps = 1
        output_steps = 1
        step_delta = lambda i: timedelta(hours=i + 1)

    elif horizon == "day":
        internal_steps = 24
        output_steps = 1
        step_delta = lambda i: timedelta(days=1)

    elif horizon == "week":
        internal_steps = 24 * 7
        output_steps = 7
        step_delta = lambda i: timedelta(days=i + 1)

    elif horizon == "month":
        internal_steps = 24 * 30
        output_steps = 30
        step_delta = lambda i: timedelta(days=i + 1)

    else:
        raise ValueError("Invalid horizon")

    # ---------------- Rolling forecast ----------------
    TIMESTEPS = 24
    X = np.array([[last_energy] * TIMESTEPS]).reshape(1, TIMESTEPS, 1)

    hourly_preds = []

    for _ in range(internal_steps):
        pred = float(model.predict(X, verbose=0)[0][0])
        hourly_preds.append(pred)

        X = np.roll(X, -1)
        X[0, -1, 0] = pred

    # ---------------- Aggregate daily values ----------------
    if horizon in ["hour", "day"]:
        base_preds = [hourly_preds[-1]]
    else:
        base_preds = []
        for i in range(output_steps):
            day_slice = hourly_preds[i * 24:(i + 1) * 24]
            base_preds.append(np.mean(day_slice))

    # ---------------- Apply device behavior ----------------
    multiplier = DEVICE_MULTIPLIER.get(device, 1.0)
    variance = DEVICE_VARIANCE.get(device, 0.05)

    final_preds = []
    for val in base_preds:
        noise = random.uniform(-variance, variance)
        adjusted = val * multiplier * (1 + noise)
        final_preds.append(round(max(adjusted, 0.001), 4))

    # ---------------- Generate dates ----------------
    now = datetime.now()
    dates = [(now + step_delta(i)).isoformat() for i in range(len(final_preds))]

    # ---------------- Smart tips (REALISTIC) ----------------
    avg = np.mean(final_preds)
    trend = "increasing" if len(final_preds) > 1 and final_preds[-1] > final_preds[0] else "stable"

    if device == "AC":
        tip = (
            "🌡 AC consumption varies daily with temperature. "
            "Using sleep mode and setting 24–26°C can significantly reduce usage."
        )

    elif device == "Fridge":
        tip = (
            "🧊 Fridge energy fluctuates with door usage. "
            "Avoid frequent opening and keep proper spacing for airflow."
        )

    elif device == "Washer":
        tip = (
            "🧺 Washing frequency impacts energy trends. "
            "Running full loads with cold water is more efficient."
        )

    elif device == "TV":
        tip = (
            "📺 Daily TV usage often varies by routine. "
            "Turning off standby power helps reduce unnecessary consumption."
        )

    elif device == "Light":
        tip = (
            "💡 Lighting demand fluctuates with daylight hours. "
            "Using LED bulbs saves up to 80% energy."
        )

    else:
        tip = "⚡ Monitoring daily trends helps identify energy-saving opportunities."

    return dates, final_preds, tip
