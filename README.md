# ⚡ Smart Energy Consumption Analysis and Prediction
Using Machine Learning with Device-Level Insights

---

#📌 Project Overview

Smart Energy Consumption Analysis and Prediction is a machine learning–powered web application designed to analyze household energy usage and predict future consumption at a device level.

The project helps users understand energy patterns, reduce wastage, and make data-driven decisions for efficient energy management.

This system combines data analysis, ML models, and a Flask-based web interface to deliver real-time insights and predictions.

---

#🎯 Key Objectives

Analyze historical smart home energy consumption data

Predict future energy usage using trained ML models

Provide device-level insights to identify high-consumption appliances

Build a clean and simple web interface for interaction

Enable scalable deployment using a modular backend design

---

#🛠️ Tech Stack

💻 Backend

Python

Flask – Web framework

NumPy, Pandas – Data processing

Scikit-Learn / Deep Learning Models – Prediction engine

---

#📊 Machine Learning

Feature engineering on time-series energy data

Trained models stored in /models

Prediction pipeline integrated into Flask API

---

#🌐 Frontend

HTML

CSS

JavaScript

Jinja2 templates (Flask)

##📁 Other Tools

Git & GitHub for version control

Jupyter Notebooks for experimentation

Virtual Environment (venv)

---

#📂 Project Structure
smart_energy_api/
```│
├── app.py                 # Main Flask application
├── requirements.txt       # Project dependencies
├── .gitignore             # Ignored files and folders
│
├── data/                  # Sample / placeholder data (datasets ignored)
│
├── models/                # Trained ML models
│
├── utils/                 # Helper functions and utilities
│
├── templates/             # HTML templates
│
├── static/                # CSS, JS, assets
│
├── colab_ntbks/            # Jupyter notebooks (experiments & training)
│
└── venv/                  # Virtual environment (ignored in Git)
```

---

#🚀 Features

📈 Energy consumption trend analysis

🔮 Future energy usage prediction

🔌 Device-level energy breakdown

🌐 Web-based interface for easy interaction

🧠 Pre-trained ML models for fast inference

---

#⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/springboardmentor110525/Smart-Energy-Consumption-Analysis-and-Prediction-using-Machine-Learning-with-Device-Level-Insights-.git
cd smart_energy_api

2️⃣ Create Virtual Environment

python -m venv venv

Activate it:

venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

▶️ Run the Application

python app.py

Then open your browser and visit:

http://127.0.0.1:5000/

---

📊 Dataset Information

kaggle dataset : https://www.kaggle.com/datasets/drmtya/smart-home-energy-consumption-optimization

Model training notebooks are available in /colab_ntbks

---

#🧠 Machine Learning Workflow

Data preprocessing and cleaning

Feature extraction and selection

Model training and evaluation

Model serialization and storage

---

##🤝 Contributors

Shaik Sameera 

Mentorship & Review by Springboard Mentor Team

