# ⚡ Smart Home Energy Consumption Forecasting

A Flask-based web application that predicts **future energy consumption** for smart home devices using a **deep learning time-series model**.  

---


## Demo

[![Preview Video](./static/preview/smart_home_energy_demo.gif)](./static/preview/smart_home_energy_demo.mp4)

---

## 🚀 What This Does

- Predicts **energy usage over time** for a selected smart device  
- Supports multiple prediction ranges:
  - **Next 24 hours**
  - **Next 7 days**
  - **Next 30 days**
- Uses historical data + temporal context to forecast future consumption
- Displays predictions alongside **timestamps**

---

## 🧠 Model Overview

- **Architecture**: Sequence-based deep learning model (trained on sliding windows)
- **Training shape**: (x, 12, 30)
- `12` → timesteps per sequence  
- `30` → engineered features per timestep

- **Custom loss function**: `asymmetric_huber`
- Penalizes under-predictions more than over-predictions

---

## 🏗️ Tech Stack

### Backend
- Flask
- TensorFlow / Keras
- Pandas, NumPy
- scikit-learn
- joblib

### ML Pipeline
- OneHotEncoder (categorical features)
- MinMaxScaler (numerical features)
- Preprocessing artifacts saved and reused at inference

### Frontend
- Jinja2 templates
- Bootstrap component for responsiveness
- CSS handled separately

---

## 📁 Project Structure
WEB/   
├─ .venv/   
├─ database/   
│  └─ sample.csv   
├─ models/   
│  └─ FinalModel.keras   
├─ node_modules/   
├─ pipeline/   
│  ├─ __pycache__/   
│  ├─ artifacts/   
│  │  ├─ feature_order.joblib   
│  │  ├─ scaler.joblib   
│  │  └─ ohe.joblib   
│  ├─ aggregation.py   
│  ├─ custom_loss.py   
│  ├─ inference.py   
│  └─ preprocess.py   
├─ static/   
│  ├─ css/   
│  │  └─ styles.css   
│  ├─ js/   
│  │  └─ chart.js   
│  ├─ preview/           # demo video and GIF   
│  └─ favicon.jpg   
├─ templates/   
│  ├─ index.html   
│  └─ main.html   
├─ utils/   
│  ├─ __pycache__/   
│  ├─ helpers.py   
│  └─ smart_tip.py   
├─ .env   
├─ .gitignore   
├─ app.py    
├─ package.json   
├─ package-lock.json   
├─ README.md   
└─ requirements.txt   

---

## 🧩 Features Used by the Model

### Binary / Integer
- `user_present`
- `status`
- `is_weekend`
- `hour_of_day`
- `day_of_week`
- `month_of_year`

### Numerical
- `indoor_temp`
- `outdoor_temp`
- `humidity`
- `light_level`
- Rolling energy statistics
- Lag features (`1H`, `1D`, `1W`)

### Categorical (One-Hot Encoded)
- `device_type`
- `room`
- `activity`

All features are **forced to match training dtypes** during inference.

---

## 🔮 Prediction Flow

1. User selects:
   - Home ID
   - Device type
   - Prediction range
2. App:
   - Generates future timestamps
   - Reuses last known categorical and numerical values
   - Builds temporal features
   - Applies saved encoders and scalers
   - Reshapes into `(1, n_steps, features)`
3. Model predicts energy consumption
4. Results displayed in a dashboard form.
