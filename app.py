from flask import Flask, render_template, request, session
import pandas as pd
import tensorflow as tf
import os
from dotenv import load_dotenv
from pipeline.custom_loss import asymmetric_huber
from pipeline.inference import make_input_seq
from pipeline.aggregation import aggregate_predictions
from utils.smart_tip import generate_smart_tip
from utils.reports import generate_csv, generate_pdf

load_dotenv()

model = tf.keras.models.load_model("./models/FinalModel.keras", custom_objects={"asymmetric_huber": asymmetric_huber}
)

df = pd.read_csv("./database/sample.csv")

RANGE_TO_STEPS = {
    "day_range": (24 * 4),
    "week_range": (7 * 24 * 4),
    "month_range": (30 * 24 * 4)
}

AGG_MAP = {
    "day_range": ["hourly"],
    "week_range": ["daily"],
    "month_range": ["daily", "weekly"]
}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        home_id = request.form['home_id']
        print(f"Logged in with Home ID: {home_id}")
        session['home_id'] = int(home_id)  
        session['logged_in'] = True
    return render_template('hist.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        device_types = request.form.getlist('device_type[]')
        pred_range = request.form['prediction_range']
        
        print(f"Device: {device_types}, Range: {pred_range}")

        n_steps = RANGE_TO_STEPS[pred_range]
        
        X = make_input_seq(df, home_id=session['home_id'], device_types=device_types,
        n_steps=n_steps)

        y_pred = model.predict(X).flatten()

        pred_by_device = {}

        idx = 0
        for device in device_types:
            pred_by_device[device] = y_pred[idx: idx + n_steps].tolist()
            idx += n_steps

        agg_modes = AGG_MAP[pred_range]
        series = {}

        for device, preds in pred_by_device.items():
            series[device] = {}

            for mode in agg_modes:
                series[device][mode] = aggregate_predictions(preds, mode).tolist()


        per_device_totals = {
            device: sum(preds)
            for device, preds in pred_by_device.items()
        }

        total_energy = sum(per_device_totals.values())

        dashboard = {
            "metadata": {
                "devices": device_types,
                "range": pred_range
            },
            "kpis":{
                "total_energy": round(total_energy, 2),
                "per_device": per_device_totals
            },
            "series": series,
            "smartTip": generate_smart_tip(per_device_totals)
        }
        print(series)
        return render_template("main.html", dashboard=dashboard)
    return render_template("main.html", dashboard=None)

@app.route('/history-dashboard', methods=['GET', 'POST'])
def hist_dash():
    if request.method == 'POST':
        hist_range = request.form['history_range']
        n_steps = RANGE_TO_STEPS[hist_range]
        agg_modes = AGG_MAP[hist_range]

        home_id = session['home_id']

        df_hist = df[df['home_id'] == home_id].tail((n_steps * 5)).copy()
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        series = {}

        for device, dfd in df_hist.groupby("device_type"):
            dfd= dfd.set_index("timestamp")
            series[device] = {}

            if "hourly" in agg_modes:
                hourly = (
                    dfd["energy_kWh"]
                    .resample("1H")
                    .sum()
                    .tail(24)
                    .tolist()
                )
                series[device]["hourly"] = hourly

            if "daily" in agg_modes:
                daily = (
                    dfd["energy_kWh"]
                    .resample("1D")
                    .sum()
                    .tail(7 if hist_range == "week_range" else 30)
                    .tolist()
                )
                series[device]["daily"] = daily

            if "weekly" in agg_modes:
                weekly = (
                    dfd["energy_kWh"]
                    .resample("1W")
                    .sum()
                    .tail(4)
                    .tolist()
                )
                series[device]["weekly"] = weekly

        total_energy = df_hist["energy_kWh"].sum()
        peak = df_hist.groupby("timestamp")["energy_kWh"].sum().max()
        per_device = (
            df_hist
            .groupby("device_type")["energy_kWh"]
            .sum()
            .sort_values(ascending=False)
        )

        top_devices = per_device.head(3)
        pie_data = per_device

        dashboard = {
            "metadata": {
                "range": hist_range,
                "points": n_steps
            },
            "kpis": {
                "total_energy": round(total_energy, 2),
                "peak": round(peak, 2)
            },
            "series": series,
            "devices": {
                "pie": pie_data.to_dict(),
                "top": top_devices.to_dict()
            }
        }

        print(dashboard)
        return render_template('hist.html', dashboard=dashboard)
    return render_template('hist.html', dashboard=None)

@app.route('/history-reports', methods=['GET', 'POST'])
def hist_rep():
    if request.method == 'POST':
        device_types = request.form.getlist('device_type[]')
        history_range = request.form['history_range']
        report_type = request.form['report_type']

        home_id = session["home_id"]

        if not device_types:
            return "No device selected", 400
        
        n_steps = RANGE_TO_STEPS[history_range]

        df_hist = (
            df[df["home_id"] == home_id]
            .tail(n_steps * 5)
            .copy()
        )

        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist = df_hist[df_hist["device_type"].isin(device_types)]

        report_df = (
            df_hist
            .groupby(["device_type", pd.Grouper(key="timestamp", freq="1h")])
            ["energy_kWh"]
            .sum()
            .reset_index()
        )

        if report_type == "csv":
            return generate_csv(report_df, device_types, history_range)

        elif report_type == "pdf":
            return generate_pdf(report_df, device_types, history_range)
        
        return "Invalid report type", 400

if __name__ == '__main__':
    app.run(debug=True)