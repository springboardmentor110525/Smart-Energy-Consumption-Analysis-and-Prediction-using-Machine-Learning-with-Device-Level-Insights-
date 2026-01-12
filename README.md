# 🔋 Smart Energy Consumption Analysis & Forecasting

A comprehensive machine learning project for analyzing smart home energy consumption patterns and forecasting future usage using Linear Regression and LSTM deep learning models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Overview

This project analyzes smart home energy data to detect usage patterns and forecast future consumption. It provides device-level insights and energy-saving recommendations using Machine Learning techniques.

### Key Features
- 📊 **Data Analysis & Preprocessing** - Clean and resample high-frequency sensor data
- 🤖 **ML Forecasting** - Predict energy usage with Linear Regression & LSTM models
- 📈 **Interactive Dashboard** - Real-time visualization of consumption trends
- 💡 **Smart Suggestions** - AI-powered energy-saving recommendations
- 🔌 **Device-Level Insights** - Analyze consumption per appliance

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML/DL** | TensorFlow, Keras, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Chart.js |
| **Web Framework** | Flask, Flask-CORS |
| **Frontend** | HTML5, CSS3, JavaScript |

## 📁 Project Structure

```
Smart-Energy-Consumption-Analysis/
├── app.py                      # Flask API & Dashboard server
├── preprocess_full_data.py     # Data preprocessing pipeline
├── Energy_Analysis.py          # Initial EDA script
├── Energy_Analysis.ipynb       # Jupyter notebook for analysis
│
├── src/                        # Source modules
│   ├── train_baseline.py       # Linear Regression training
│   ├── train_lstm.py           # LSTM model training
│   ├── evaluate_models.py      # Model evaluation & comparison
│   └── utils.py                # Utility functions
│
├── models/                     # Trained models
│   ├── linear_regression.pkl   # Baseline model
│   └── lstm_model.keras        # LSTM deep learning model
│
├── templates/                  # HTML templates
│   └── dashboard.html          # Main dashboard UI
│
├── static/                     # Static assets
│   ├── css/                    # Stylesheets
│   └── js/                     # JavaScript files
│
├── results/                    # Model outputs & metrics
├── *.csv                       # Processed data files
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/deekshith1818/Smart-Energy-Consumption-Analysis.git
   cd Smart-Energy-Consumption-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow flask flask-cors
   ```

3. **Download the dataset**
   - Download `HomeC.csv` from the [Smart Home Energy Dataset](https://www.kaggle.com/datasets)
   - Place it in the project root directory

### Running the Application

1. **Preprocess the data** (if starting fresh)
   ```bash
   python preprocess_full_data.py
   ```

2. **Train the models** (optional - pre-trained models included)
   ```bash
   python src/train_baseline.py
   python src/train_lstm.py
   ```

3. **Start the Dashboard**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/overview` | GET | Energy consumption overview |
| `/api/devices` | GET | Device-wise consumption data |
| `/api/hourly` | GET | Hourly consumption patterns |
| `/api/daily` | GET | Daily consumption trends |
| `/api/suggestions` | GET | Energy-saving recommendations |
| `/api/predict` | POST | Make consumption predictions |
| `/api/model-comparison` | GET | Model performance metrics |

## 🧠 Models

### Linear Regression (Baseline)
- Feature-based regression model
- Fast inference for real-time predictions
- Good for linear consumption patterns

### LSTM Neural Network
- Sequence-based deep learning model
- Captures temporal dependencies
- Better for complex consumption patterns

## 📈 Project Progress

- [x] Data Collection & Cleaning
- [x] Exploratory Data Analysis (EDA)
- [x] Feature Engineering
- [x] Baseline Model (Linear Regression)
- [x] LSTM Deep Learning Model
- [x] Model Evaluation & Comparison
- [x] Flask REST API
- [x] Interactive Dashboard
- [x] Smart Suggestions Engine

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Deekshith** - [GitHub](https://github.com/deekshith1818)

---

⭐ Star this repository if you found it helpful!
