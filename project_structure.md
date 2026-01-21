# Smart Energy Consumption Analysis - Project Structure

## Root Directory
- **`app.py`**: The main Flask application entry point. Handles web routes, API endpoints, and orchestrates the dashboard.
- **`Energy_Analysis.ipynb`**: Jupyter notebook containing the initial data analysis, visualization, and model experimentation.
- **`Energy_Analysis.py`**: Python script version of the analysis, likely for automation or pipeline usage.
- **`auto_train.py`**: Script to automate the training of models (likely both baseline and LSTM), keeping them up-to-date with new data.
- **`preprocess_full_data.py`**: Script for data cleaning and preprocessing pipelines.
- **`generate_2025_data.py`**: Utility to generate synthetic future data for testing predictions.

## Data Files
- **`HomeC.csv`**: The raw, original dataset containing detailed home energy consumption data.
- **`clean_energy_data.csv`** & **`clean_energy_data_2025.csv`**: Processed datasets ready for analysis and training.
- **`scaled_energy_data.csv`**: Normalized/scaled data used for neural network training (LSTM).
- **`train_data.csv`, `test_data.csv`**: Split datasets for model training and validation.
- **`train_features.csv`, `test_features.csv`**: Feature matrices for model input.

## Source Code (`src/`)
- **`train_baseline.py`**: Training logic for the Linear Regression baseline model.
- **`train_lstm.py`**: Training logic for the LSTM (Long Short-Term Memory) deep learning model.
- **`evaluate_models.py`**: Scripts to assess model performance (MAE, RMSE, R²).
- **`utils.py`**: Shared utility functions for data manipulation and common tasks.

## Web Application (`static/` & `templates/`)
### Templates (HTML)
- **`dashboard.html`**: Main landing page with high-level metrics and charts.
- **`predictions.html`**: Interface for running model predictions (Linear vs LSTM).
- **`devices.html`**: Detailed breakdown of energy usage by specific devices.
- **`history.html`**: Historical data analysis and trends.
- **`reports.html`**: Tool to generate and download consumption reports.
- **`settings.html`**: Configuration page for dataset management and model retraining.

### Static Assets
- **`css/style.css`**: Main stylesheet (recently updated with responsive sidebar).
- **`js/`**: JavaScript files corresponding to each page (`dashboard.js`, `sidebar.js`, etc.) handling frontend logic and interactivity.

## Models (`models/`)
- **`linear_regression.pkl`**: Saved Scikit-Learn model object.
- **`lstm_model.keras`**: Saved Keras/TensorFlow deep learning model.

## Results (`results/`)
- Stores generated plots, metrics, or logs from training runs.
