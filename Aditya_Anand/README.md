# Smart Energy Consumption Analysis & Forecasting

## Project Overview
This project analyzes smart home energy data to detect usage patterns and forecast future consumption. It aims to reduce energy wastage and provide device-level insights using Machine Learning.

## Objectives
- **Data Analysis**: Clean and resample high-frequency sensor data.
- **Forecasting**: Predict future energy usage using LSTM (Deep Learning) and Linear Regression.
- **Dashboard**: Visualize trends and anomalies (Outliers) through an interactive Web Application.

## Tech Stack
- **Language**: Python
- **Libraries**: Pandas, NumPy, Seaborn, Matplotlib, TensorFlow/Keras, Flask, Scikit-learn
- **Dataset**: SmartHome Energy Monitoring Dataset (`HomeC_augmented.csv`)

## Current Progress (Milestone 1 & 2)
- [x] Data Collection & Cleaning
- [x] Handling Null Values & Duplicates
- [x] Exploratory Data Analysis (EDA)
- [x] Outlier Detection & Distribution Plots
- [x] Resampling Data to Hourly Intervals
- [x] Model Training (Baseline & LSTM)
- [x] Web Dashboard Deployment

## How to Run
1. **Clone the repository**.
2. **Ensure the dataset** `HomeC_augmented.csv` is in the same directory.
3. **Open in Jupyter Notebook**:
   - Open `Aditya_Anand.ipynb` to view the analysis and run the cells.
4. **Run the Web Application** (Optional):
   - Install requirements: `pip install -r requirements.txt`
   - Run the app: `python app.py`
   - Access at: `http://localhost:5000`
