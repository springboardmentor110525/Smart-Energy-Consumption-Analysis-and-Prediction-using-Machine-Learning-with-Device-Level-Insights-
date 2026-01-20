from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

print("Loading Robust Model...")
# Ensure you have 'website_model.h5', 'scaler_X.pkl', and 'scaler_y.pkl'
try:
    model = tf.keras.models.load_model('website_model.h5', compile=False)
    scaler_X = joblib.load('scaler_X.pkl') 
    scaler_y = joblib.load('scaler_y.pkl')
    print("✅ Model & Scalers Loaded Successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")
    print("Make sure you ran the Jupyter Notebook code to save scaler_X and scaler_y!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- 1. HANDLE INPUT ---
        user_input_str = request.form.get('prev_usage')
        if user_input_str is None:
             user_input_str = request.form.get('usage')
        
        if user_input_str is None or user_input_str.strip() == "":
            return render_template('index.html', error_text="Error: Input cannot be empty.")

        prev_usage = float(user_input_str)
        day_of_week = int(request.form['day_of_week']) 
        
        # --- 2. SMART SWITCH (Home vs Industrial) ---
        if prev_usage > 10.0:
            pred_next_hour = prev_usage * 0.98  # Persistence for big numbers
        else:
            # AI Prediction for Home
            features = [12.0, float(day_of_week), 15.0, 6.0, prev_usage, prev_usage, prev_usage]
            final_features = np.array([features])
            scaled_features = scaler_X.transform(final_features) # Use scaler_X if you have it, else scaler
            lstm_input = scaled_features.reshape((1, 1, 7))
            
            output = model.predict(lstm_input)[0][0]
            # Handle Inverse Transform if using Dual Scalers
            # real_prediction = scaler_y.inverse_transform([[output]]) 
            # pred_next_hour = real_prediction[0][0]
            
            # If using Single Scaler (Lite Model standard), the output is already the value
            pred_next_hour = max(0, output) 

        # --- 3. PROJECTIONS ---
        pred_3_hours = pred_next_hour * 3
        pred_next_week = pred_next_hour * 24 * 7
        pred_next_month = pred_next_hour * 24 * 30

        # --- 4. DETAILED SUGGESTIONS LOGIC (The New Part) ---
        suggestions = []
        suggestion_color = "green"

        # Case A: Industrial / Huge Input
        if prev_usage > 10.0:
             suggestions = [
                 "🏭 **Industrial Load Detected:** Exceeds residential limits.",
                 "⚠️ **Peak Demand Warning:** Current load contributes to peak grid stress.",
                 "⚙️ **Optimization:** Inspect heavy machinery or HVAC cooling towers.",
                 "📉 **Action:** Shift high-load operations to off-peak hours (after 9 PM)."
             ]
             suggestion_color = "#b91c1c" # Dark Red

        # Case B: High Home Usage (> 2.0 kW)
        elif pred_next_hour > 2.0:
             suggestions = [
                 "🚨 **High Usage Alert:** Consumption is significantly above average.",
                 "❄️ **AC/Heater:** It seems the HVAC system is running at full power. Consider raising the thermostat.",
                 "🔌 **Phantom Load:** Unplug devices not in use (Microwaves, Gaming Consoles).",
                 "⏳ **Delay:** Wait until evening to run the Dishwasher or Washing Machine."
             ]
             suggestion_color = "#dc2626" # Red

        # Case C: Moderate Usage (1.0 - 2.0 kW)
        elif pred_next_hour > 1.0:
             suggestions = [
                 "⚠️ **Moderate Usage:** Slightly higher than ideal.",
                 "💡 **Lighting:** Check if lights are left on in empty rooms.",
                 "💻 **Home Office:** Computers/Monitors might be drawing excess power.",
                 "🌡️ **Water Heater:** Ensure the water heater isn't running continuously."
             ]
             suggestion_color = "#d97706" # Orange

        # Case D: Low/Efficient Usage (< 1.0 kW)
        else:
             suggestions = [
                 "✅ **Excellent Efficiency:** Your usage is optimal.",
                 "🌿 **Eco-Friendly:** You are in the green zone!",
                 "🧺 **Recommendation:** Great time to run one major appliance if needed.",
                 "🔋 **Status:** Battery/Solar storage (if available) is maximizing savings."
             ]
             suggestion_color = "#16a34a" # Green

        # --- 5. CHARTS DATA ---
        device_names = ['Furnace', 'Fridge', 'AC', 'LivingRoom', 'Microwave', 'Office']
        device_values = [
            float(pred_next_hour * 0.35), 
            float(pred_next_hour * 0.20), 
            float(pred_next_hour * 0.15), 
            float(pred_next_hour * 0.10), 
            float(pred_next_hour * 0.10), 
            float(pred_next_hour * 0.10)
        ]

        return render_template('index.html', 
                               p_1h=f'{pred_next_hour:.4f}',
                               p_3h=f'{pred_3_hours:.2f}',
                               p_week=f'{pred_next_week:.2f}',
                               p_month=f'{pred_next_month:.2f}',
                               suggestions_list=suggestions, # Sending the LIST now
                               s_color=suggestion_color,
                               device_names=device_names,
                               device_values=device_values,
                               show_result=True)

    except Exception as e:
        print(f"SERVER ERROR: {e}") 
        return render_template('index.html', error_text=f'Server Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, port=5000)