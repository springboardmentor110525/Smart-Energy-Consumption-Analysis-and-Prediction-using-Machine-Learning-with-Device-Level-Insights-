from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# ------------------------------------
# LOAD CLEANED DATA
# ------------------------------------
df = pd.read_csv("data/House_1_cleaned_named.csv")
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

APPLIANCES = [
    'Fridge', 'Freezer', 'Washing_Machine', 'Dishwasher',
    'Computer', 'Television', 'Microwave', 'Kettle', 'Toaster'
]

# ------------------------------------
# SMART ENERGY SUGGESTIONS
# ------------------------------------
SMART_SUGGESTIONS = {
    "Washing_Machine": [
        "Run the washing machine during off-peak hours (late night or early morning).",
        "Avoid half-load washing; always use full loads.",
        "Use cold water or eco mode to reduce heating energy.",
        "Reduce washing frequency by batching clothes together.",
        "Prefer shorter wash cycles when possible."
    ]
}

# ------------------------------------
# ESTIMATED ENERGY SAVINGS
# ------------------------------------
ESTIMATED_SAVINGS = {
    "Washing_Machine":
        "Estimated energy reduction of 15–25% per week by using eco modes, "
        "full loads, and off-peak operation."
}

# ------------------------------------
# DASHBOARD ROUTE
# ------------------------------------
@app.route("/")
def dashboard():

    # ---- Hourly Aggregate Usage (Last 24 hours)
    hourly = df['Aggregate'].resample('H').mean().tail(24)
    labels = ",".join(hourly.index.strftime('%H:%M'))
    values = ",".join(map(str, hourly.values))

    # ---- Appliance-wise Total Usage
    appliance_usage = df[APPLIANCES].sum()
    appliance_labels = ",".join(appliance_usage.index)
    appliance_values = ",".join(map(str, appliance_usage.values))

    # ---- Top Appliance Insights
    top_appliance = appliance_usage.idxmax()
    highest_usage = round(float(appliance_usage.max()), 2)
    peak_hour = df['Aggregate'].resample('H').mean().idxmax().strftime('%H:%M')

    # ---- Washing Machine Suggestions
    washing_suggestions = SMART_SUGGESTIONS.get("Washing_Machine", [])
    washing_savings = ESTIMATED_SAVINGS.get("Washing_Machine")

    return render_template(
        "index.html",
        labels=labels,
        values=values,
        appliance_labels=appliance_labels,
        appliance_values=appliance_values,
        top_appliance=top_appliance,
        highest_usage=highest_usage,
        peak_hour=peak_hour,
        washing_suggestions=washing_suggestions,
        washing_savings=washing_savings
    )

# ------------------------------------
# PREDICTION ROUTE
# ------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    predictions = []
    labels = []
    selected_appliance = None
    selected_horizon = None
    prediction_insight = None

    if request.method == "POST":
        selected_appliance = request.form["appliance"]
        selected_horizon = request.form["horizon"]

        series = df[selected_appliance]

        # ---- NEXT HOURS (6 HOURS)
        if selected_horizon == "hours":
            avg = series.resample("H").mean().tail(24).mean()
            labels = [f"H+{i}" for i in range(1, 7)]
            predictions = [
                round(avg * (1 + np.sin(i / 2) * 0.1), 2)
                for i in range(1, 7)
            ]

        # ---- NEXT DAY
        elif selected_horizon == "day":
            avg = series.resample("D").mean().tail(7).mean()
            labels = ["Next Day"]
            predictions = [round(avg, 2)]

        # ---- NEXT WEEK
        elif selected_horizon == "week":
            avg = series.resample("D").mean().tail(7).mean()
            labels = [f"Day {i}" for i in range(1, 8)]
            predictions = [
                round(avg * (1 + np.sin(i / 3) * 0.1), 2)
                for i in range(1, 8)
            ]

        # ---- NEXT TWO MONTHS
        elif selected_horizon == "months":
            avg = series.resample("M").mean().tail(2).mean()
            labels = ["Month 1", "Month 2"]
            predictions = [
                round(avg * (1 + np.sin(i) * 0.1), 2)
                for i in range(1, 3)
            ]

        # ---- INSIGHT TEXT
        max_value = max(predictions)
        max_index = predictions.index(max_value)
        peak_time = labels[max_index]

        prediction_insight = (
            f"⚡ High energy consumption of approximately "
            f"{max_value} Wh is expected around {peak_time}."
        )

    return render_template(
        "predict.html",
        appliances=APPLIANCES,
        predictions=predictions,
        labels=labels,
        selected_appliance=selected_appliance,
        selected_horizon=selected_horizon,
        prediction_insight=prediction_insight
    )

# ------------------------------------
# COMPARISON ROUTE
# ------------------------------------
@app.route("/compare")
def compare():
    appliance_cols = [
        "Fridge","Freezer","Washing_Machine","Dishwasher",
        "Computer","Television","Microwave","Kettle","Toaster"
    ]

    totals = df[appliance_cols].sum().sort_values(ascending=False)

    labels = totals.index.tolist()
    values = [float(v) for v in totals.values]

    return render_template(
        "compare.html",
        labels=labels,
        values=values
    )

# ------------------------------------
# RUN APP
# ------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
