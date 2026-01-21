---

# 🔌 Smart Energy Consumption Analysis — House 1 (REFIT Dataset)

This project presents a **complete data-driven analysis and forecasting system for household energy consumption** using the **REFIT Smart Home Dataset (House 1)**.
The work covers **data preprocessing, analysis, modeling, and an interactive dashboard built with Flask**.

---

## 📌 Project Overview

The goal of this project is to:

* Understand **appliance-level energy usage patterns**
* Identify **high-consuming appliances and peak usage periods**
* Build **forecasting models** for future energy consumption
* Present insights through a **clear, interactive dashboard**

This project emphasizes **data understanding and visualization first**, followed by **prediction and insights**, which aligns with real-world analytics workflows.

---

## 🎯 Project Objectives

* Analyze household energy consumption at appliance level
* Clean and preprocess large-scale time-series energy data
* Engineer meaningful time-based and statistical features
* Build and evaluate a **baseline Linear Regression model**
* Develop an **LSTM-based time-series forecasting model**
* Evaluate models using MAE, RMSE, and R²
* Deploy results using a **Flask-based web dashboard**

---

## 📁 Project Structure

```
SmartEnergyProject/
│
├── data/
│   ├── House_1_cleaned_named.csv      # Cleaned dataset with real appliance names
│   └── README.md                      # Dataset description
│
├── notebooks/
│   ├── 01_Data_Analysis.ipynb         # EDA and preprocessing
│   ├── 02_Feature_Engineering.ipynb   # Feature creation
│   ├── 03_Baseline_Model.ipynb        # Linear Regression + cross-validation
│   ├── 04_LSTM_Model.ipynb            # LSTM model training & evaluation
│   └── 05_Dashboard_Visualization.ipynb
│
├── app.py                             # Flask application
│
├── templates/
│   ├── index.html                     # Dashboard UI
│   ├── predict.html                   # Prediction page
│   └── compare.html                   # Appliance comparison page
│
├── static/
│   └── style.css                      # Dashboard styling
│
├── README.md                          # Project documentation
└── .gitignore                         # Ignore venv, cache, model files
```

---

## 🧹 Module 1 & 2: Data Cleaning and Preprocessing

### Steps Performed

* Loaded REFIT House 1 dataset (~6.9 million rows)
* Verified data quality:

  * No missing values
  * No duplicate records
* Renamed appliance columns to **real appliance names**:

  * Fridge, Freezer, Washing Machine, Dishwasher, etc.
* Converted timestamps to `datetime`
* Set time column as index for time-series analysis
* Created `active_count` feature (number of active appliances)
* Filtered rows with **active_count ≥ 3**
* Resampled data:

  * Hourly
  * Daily
* Normalized numerical values using **Min-Max Scaling**
* Split data into:

  * Training (70%)
  * Validation (15%)
  * Testing (15%)

---

## 📊 Module 1: Exploratory Data Analysis (EDA)

### Analysis Includes

* Summary statistics
* Distribution plots for:

  * Aggregate energy usage
  * Individual appliances
* Boxplots for outlier inspection
* Correlation heatmap showing relationships between appliances and total energy consumption

All plots and tables are available inside the notebook.

---

## 🧠 Module 3: Feature Engineering

### Engineered Features

* **Time-based features**

  * Hour
  * Day
  * Weekday
  * Month
* **Appliance aggregation features**

  * Total appliance load
  * Mean appliance load
  * Maximum appliance load
* **Lag features**

  * Previous hour (`lag1`)
  * Previous day (`lag24`)
* **Rolling statistics**

  * 3-hour rolling mean
  * 24-hour rolling mean

Final dataset:

* 3702 samples
* 21 features
* 1 target variable (aggregate energy consumption)

---

## 📈 Module 4: Baseline Model Development

### Model Used

* **Linear Regression**

### Evaluation

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

### Overfitting Check

* Applied **TimeSeriesSplit cross-validation**
* Observed stable MAE across folds
* Conclusion: **Baseline model does not overfit**

---

## 🤖 Module 5: LSTM Model Development

### Model Architecture

* LSTM layer (64 units)
* Dropout for regularization
* Dense output layer
* Lookback window: **24 hours**

### Training & Evaluation

* Optimizer: Adam
* Loss: Mean Squared Error
* Evaluated using MAE, RMSE, and R²
* Compared against Linear Regression using the same test set

---

## 🔗 Module 6: Model Evaluation and Integration

* Evaluated both models using:

  * MAE
  * RMSE
  * R² score
* Saved the trained LSTM model in `.keras` format
* Built a **Flask-compatible prediction function**
* Verified predictions using real historical data samples

---

## 🖥️ Module 7: Dashboard and Visualization (Flask)

### Dashboard Features

* Flask-based web application
* Real-time data-driven dashboard
* Visualizations include:

  * Hourly energy consumption (bar chart)
  * Appliance-wise energy distribution (pie chart)
  * Appliance comparison
* Prediction portal:

  * Forecasts next hours, days, and weeks
  * Generates readable insights such as peak usage times
* Designed with clean, responsive UI using HTML & CSS

This module makes the project **demonstration-ready and mentor-friendly**.

---

## 🗂 Dataset Information

* Dataset: REFIT Smart Home Energy Dataset — House 1
* Original size: ~6.9 million rows
* After filtering: ~2.7 million rows
* Uploaded dataset: cleaned and reduced for GitHub compatibility

---

## 🚀 Future Enhancements

* Appliance-specific forecasting models
* Advanced LSTM architectures
* Cloud deployment
* Smarter rule-based energy-saving recommendations
* Interactive dashboards with more user controls

---

## 👤 Author

**Uppanda Keerthana**
Smart Energy Consumption Analysis Project

---

