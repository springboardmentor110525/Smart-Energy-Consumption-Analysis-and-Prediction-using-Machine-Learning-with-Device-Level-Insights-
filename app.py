from flask import Flask, render_template, request, session, jsonify
import pandas as pd
import tensorflow as tf
import os
from dotenv import load_dotenv
from pipeline.custom_loss import asymmetric_huber
from pipeline.inference import make_input_seq

load_dotenv()

model = tf.keras.models.load_model("./models/FinalModel.keras", custom_objects={"asymmetric_huber": asymmetric_huber}
)

df = pd.read_csv("./database/sample.csv")

PRED_RANGE_TO_STEPS = {
    "24h_hourly": (24 * 4),
    "7d_daily": (7 * 24 * 4),
    "30d_monthly": (30 * 24 * 4)
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
    return render_template('main.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        device_type = request.form['device_type']
        pred_range = request.form['prediction_range']
        
        print(f"Device: {device_type}, Range: {pred_range}")

        n_steps = PRED_RANGE_TO_STEPS[pred_range]
        
        X = make_input_seq(df, home_id=session['home_id'], device_type=device_type,
        n_steps=n_steps)

        y_pred = model.predict(X)
        print(y_pred)

        # time_list = pd.Series(time_idx).dt.strftime("%Y-%m-%d %H:%M").tolist()
        y_list = y_pred.flatten().tolist()

        # pred_time_pairs = list(zip(time_list, y_list))
        # print(pred_time_pairs)
        return render_template("main.html", predictions=y_list)


if __name__ == '__main__':
    app.run(debug=True)