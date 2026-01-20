"""
Module 5: LSTM Model Development
Week 5-6 Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

class LSTMEnergyModel:
    """
    LSTM model for energy consumption time series forecasting
    """
    
    def __init__(self, sequence_length=24):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of time steps to use for prediction
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = None
        self.predictions = None
        self.metrics = {}
        
    def prepare_sequences(self, data):
        """
        Create sequences for LSTM
        
        Args:
            data: Time series data array
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, target_col, test_size=0.2):
        """
        Prepare and scale data for LSTM
        
        Args:
            df: DataFrame with time series data
            target_col: Target variable column
            test_size: Test set proportion
        """
        print("\n[INFO] Preparing data for LSTM model...")
        
        # Extract target column
        data = df[target_col].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.prepare_sequences(scaled_data)
        
        # Reshape X for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split into train and test
        split_idx = int(len(X) * (1 - test_size))
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        print(f"   [OK] Sequence length: {self.sequence_length}")
        print(f"   [OK] Training sequences: {len(self.X_train):,}")
        print(f"   [OK] Test sequences: {len(self.X_test):,}")
        print(f"   [OK] X shape: {self.X_train.shape}")
        print(f"   [OK] y shape: {self.y_train.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self, units=128, dropout=0.2, learning_rate=0.001):
        """
        Build LSTM architecture
        
        Args:
            units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        print("\n[INFO] Building LSTM model architecture...")
        
        self.model = Sequential([
            # First LSTM layer
            LSTM(units=units, return_sequences=True, 
                 input_shape=(self.sequence_length, 1)),
            Dropout(dropout),
            
            # Second LSTM layer
            LSTM(units=units//2, return_sequences=True),
            Dropout(dropout),
            
            # Third LSTM layer
            LSTM(units=units//4, return_sequences=False),
            Dropout(dropout),
            
            # Dense layers
            Dense(units=32, activation='relu'),
            Dropout(dropout/2),
            
            Dense(units=16, activation='relu'),
            
            # Output layer
            Dense(units=1)
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print("   [OK] Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train LSTM model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation data proportion
        """
        print(f"\n[INFO] Training LSTM model...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Validation split: {validation_split}")
        
        if self.model is None:
            print("   [ERROR] Please build model first!")
            return
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n   [OK] Training complete!")
        
        return self.history
    
    def predict(self):
        """Make predictions on test set"""
        print("\n[INFO] Making predictions...")
        
        if self.model is None:
            print("   [ERROR] Please train model first!")
            return
        
        # Predict
        predictions_scaled = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform to original scale
        self.predictions = self.scaler.inverse_transform(predictions_scaled)
        y_test_original = self.scaler.inverse_transform(self.y_test)
        
        print(f"   [OK] Generated {len(self.predictions):,} predictions")
        
        return self.predictions, y_test_original
    
    def evaluate(self):
        """Evaluate model performance"""
        print("\n[INFO] Evaluating LSTM model performance...")
        
        if self.predictions is None:
            print("   [ERROR] Please make predictions first!")
            return
        
        # Get original scale values
        y_test_original = self.scaler.inverse_transform(self.y_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, self.predictions)
        r2 = r2_score(y_test_original, self.predictions)
        mape = np.mean(np.abs((y_test_original - self.predictions) / y_test_original)) * 100
        
        self.metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        print("\n" + "="*50)
        print("   LSTM MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"   Mean Squared Error (MSE):  {mse:.6f}")
        print(f"   Root Mean Squared Error:   {rmse:.6f}")
        print(f"   Mean Absolute Error (MAE): {mae:.6f}")
        print(f"   R² Score:                  {r2:.6f}")
        print(f"   MAPE:                      {mape:.2f}%")
        print("="*50)
        
        return self.metrics
    
    def visualize_training(self, save_path='reports/figures/lstm_training.png'):
        """Visualize training history"""
        print("\n[INFO] Creating training visualizations...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Model Loss During Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Model MAE During Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"   [OK] Training visualization saved to {save_path}")
        plt.show()
    
    def visualize_predictions(self, save_path='reports/figures/lstm_predictions.png'):
        """Visualize predictions"""
        print("\n[INFO] Creating prediction visualizations...")
        
        y_test_original = self.scaler.inverse_transform(self.y_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Actual vs Predicted scatter
        axes[0, 0].scatter(y_test_original, self.predictions, alpha=0.5, s=10)
        axes[0, 0].plot([y_test_original.min(), y_test_original.max()],
                        [y_test_original.min(), y_test_original.max()],
                        'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('LSTM: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series comparison
        n_points = min(500, len(y_test_original))
        axes[0, 1].plot(range(n_points), y_test_original[:n_points], 
                       label='Actual', alpha=0.7, linewidth=1.5)
        axes[0, 1].plot(range(n_points), self.predictions[:n_points],
                       label='Predicted', alpha=0.7, linewidth=1.5)
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Energy Consumption')
        axes[0, 1].set_title(f'Time Series Comparison (First {n_points} points)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals
        residuals = y_test_original.flatten() - self.predictions.flatten()
        axes[1, 0].scatter(self.predictions, residuals, alpha=0.5, s=10)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residual Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error distribution
        axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Residual Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"   [OK] Prediction visualization saved to {save_path}")
        plt.show()
    
    def save_model(self, model_path='models/best_lstm_model.h5', 
                   scaler_path='models/lstm_scaler.pkl'):
        """Save model and scaler"""
        if self.model is None:
            print("   [ERROR] No model to save!")
            return
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"   [OK] Model saved to {model_path}")
        print(f"   [OK] Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='models/best_lstm_model.h5',
                   scaler_path='models/lstm_scaler.pkl'):
        """Load model and scaler"""
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"   [OK] Model loaded from {model_path}")
        print(f"   [OK] Scaler loaded from {scaler_path}")


def main():
    """Main execution function"""
    print("="*60)
    print("SMART ENERGY ANALYSIS - LSTM MODEL")
    print("="*60)
    
    # Load data
    print("\n[INFO] Loading data...")
    df = pd.read_csv('data/processed/energy_data_clean.csv',
                     index_col='time', parse_dates=True)
    print(f"   [OK] Loaded {len(df):,} records")
    
    # Initialize LSTM model
    lstm_model = LSTMEnergyModel(sequence_length=24)
    
    # Prepare data
    target_col = 'use_HO' if 'use_HO' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    print(f"\n   Target variable: {target_col}")
    
    X_train, X_test, y_train, y_test = lstm_model.prepare_data(df, target_col)
    
    # Build model
    lstm_model.build_model(units=128, dropout=0.2, learning_rate=0.001)
    
    # Train model
    history = lstm_model.train(epochs=50, batch_size=32)
    
    # Visualize training
    lstm_model.visualize_training()
    
    # Make predictions
    predictions, y_test_original = lstm_model.predict()
    
    # Evaluate model
    metrics = lstm_model.evaluate()
    
    # Visualize predictions
    lstm_model.visualize_predictions()
    
    # Save model
    lstm_model.save_model()
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('reports/results/lstm_metrics.csv', index=False)
    print(f"\n   [OK] Metrics saved to reports/results/lstm_metrics.csv")
    
    print("\n" + "="*60)
    print("[OK] LSTM MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return lstm_model, metrics


if __name__ == "__main__":
    lstm_model, metrics = main()
