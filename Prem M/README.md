# ⚡ Smart Energy Consumption Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![ML](https://img.shields.io/badge/Model-LSTM%20%2F%20TensorFlow-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Empowering Smart Homes with Real-Time Energy Intelligence.**

A full-stack Machine Learning application that forecasts home energy usage, detects anomalies, and provides device-level insights using Deep Learning (LSTM) and a modern Glassmorphism UI.

---

## 🚀 Key Features
* **🧠 Deep Learning Core:** Uses **Long Short-Term Memory (LSTM)** networks to capture complex time-series patterns.
* **🔮 Real-Time Forecasting:** Predicts energy consumption for the **Next Hour, Week, and Month**.
* **🎨 Glassmorphism UI:** A responsive, modern interface built with translucent CSS effects and dynamic animations.
* **⚡ Smart "Eco-Switch" Logic:**
    * **Residential Mode (<10kW):** Precision AI forecasting.
    * **Industrial Mode (>10kW):** Robust outlier handling system.
* **📊 Device Disaggregation:** Estimates power usage breakdown (HVAC, Kitchen, Lights) from total load.

---

## 🛠️ Tech Stack

### **Machine Learning**
* **TensorFlow & Keras:** LSTM Model training and inference.
* **Scikit-Learn:** MinMax Scaling and data preprocessing.
* **Pandas & NumPy:** Data cleaning and rolling-window feature engineering.

### **Web Application**
* **Backend:** Python Flask (REST API).
* **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript.
* **Visualization:** Chart.js (Interactive Line & Pie charts).

---

## 📂 Project Structure

```bash
Smart-Energy-Analysis/
├── app.py                  # Main Flask Application
├── website_model.h5        # Trained LSTM Model (Deep Learning)
├── scaler_X.pkl            # Input Feature Scaler
├── scaler_y.pkl            # Output Target Scaler
├── requirements.txt        # Python Dependencies
├── templates/
│   └── index.html          # Frontend UI
├── static/                 # CSS/JS files
└── notebooks/              # Jupyter Notebooks for Training

