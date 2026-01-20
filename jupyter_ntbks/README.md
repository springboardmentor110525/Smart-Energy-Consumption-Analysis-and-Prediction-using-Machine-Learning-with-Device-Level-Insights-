# Smart Energy Consumption Analysis and Prediction  
**Device-Level Forecasting with Time-Series Deep Learning**

## Overview  
This repository contains the **Jupyter Notebook–based machine learning pipeline** for a Smart Energy Consumption Analysis system.  
The primary objective is to **model, predict, and analyze energy consumption patterns at the device level** using multivariate time-series data from smart homes.

This repository focuses exclusively on:
- data preprocessing  
- feature engineering  
- sequence generation  
- model training and evaluation  

The deployed web application (Flask + dashboards + smart tips) consumes the trained model artifacts produced here.

---

## Dataset  
**Source:**  
Smart Home Energy Consumption Optimization Dataset  
https://www.kaggle.com/datasets/drmtya/smart-home-energy-consumption-optimization  

The dataset includes timestamped smart-home data such as:
- device-level usage  
- environmental conditions  
- user presence and activity  
- historical energy consumption  

---

## Problem Definition  
Energy consumption is a **time-dependent and non-linear process** influenced by multiple contextual factors.  
Static regression models struggle to capture:
- temporal dependencies  
- usage spikes  
- device behavior patterns  

This project formulates the task as a **multivariate time-series forecasting problem**, predicting energy consumption **per timestamp** rather than per sequence.

---

## Modeling Approach  

### Models Explored
- Linear Regression (baseline)
- LSTM-based Deep Learning Model (final)

### Final Model Characteristics
- Multivariate LSTM
- Input shape: `(sequence_length, number_of_features)`
- Output: energy prediction for **each timestamp**
- Globally trained (house-agnostic)
- Designed to support downstream personalization using correction factors

---

## Model Performance (Test Set)

| Metric | Linear Regression | LSTM | 
|-------------|-----------------|-----------| 
| Test MAE | <span style="color:red">0.016</span> | <span style="color:green">0.003</span> | 
| Test RMSE | <span style="color:red">0.025</span> | <span style="color:green">0.008</span> | 
| Test R² | <span style="color:red">0.70</span> | <span style="color:green">0.96</span> |

The LSTM significantly outperforms the baseline by learning:
- temporal relationships  
- device interaction effects  
- peak consumption behavior  

---

## Feature Engineering Summary  
The final feature set includes:
- temporal indicators (hour of day, day of week, month, weekend flag)
- environmental variables (indoor temperature, outdoor temperature, humidity, light level)
- device indicators (one-hot encoded)
- occupancy and activity indicators
- historical context features during training (lags and rolling statistics)

Cyclic encodings were evaluated but excluded due to reduced empirical performance.

---

## Data Splitting and Sequence Generation  
- Train / Validation / Test split  
- Sequences generated **after preprocessing** to avoid data leakage  
- Targets aligned per timestamp for sequence-to-sequence learning  

---

## Folder Structure

(will be added soon)

---

## Tools & Libraries  
- Python 3.x  
- NumPy  
- Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  
- Joblib  
- Jupyter Notebook  

---

## Notes  
- This repository contains **only the ML pipeline** (milestone 1 - 6)
- Dashboards, aggregation, visualization, and smart tips are handled in the web application layer
- The trained model is API-ready and optimized for deployment

---

## Status  
- Data preprocessing completed  
- Feature engineering finalized  
- LSTM model trained and evaluated  
- Model artifacts saved for deployment