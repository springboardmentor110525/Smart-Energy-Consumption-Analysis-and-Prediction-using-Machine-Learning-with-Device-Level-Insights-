"""
Module 4: Baseline Model Development - Linear Regression
Week 3-4 Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class BaselineModel:
    """
    Baseline Linear Regression model for energy consumption prediction
    """
    
    def __init__(self):
        """Initialize baseline model"""
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.metrics = {}
        
    def prepare_data(self, df, target_col, test_size=0.2, random_state=42):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features
            target_col: Target variable column name
            test_size: Test set proportion
            random_state: Random seed
        """
        print("\n📊 Preparing data for baseline model...")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Select only numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Handle any remaining NaN values
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        print(f"   ✓ Training set: {len(self.X_train):,} samples")
        print(f"   ✓ Test set: {len(self.X_test):,} samples")
        print(f"   ✓ Features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self):
        """Train the baseline model"""
        print("\n🎯 Training Linear Regression model...")
        
        if self.X_train is None:
            print("   ✗ Please prepare data first!")
            return
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        print("   ✓ Model trained successfully!")
        
        # Get feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'coefficient': abs(self.model.coef_)
        }).sort_values('coefficient', ascending=False)
        
        print(f"\n   Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"      {row['feature']}: {row['coefficient']:.4f}")
        
        return self.model
    
    def predict(self):
        """Make predictions on test set"""
        print("\n🔮 Making predictions...")
        
        if self.model is None:
            print("   ✗ Please train model first!")
            return
        
        # Predict on test set
        self.predictions = self.model.predict(self.X_test)
        
        print(f"   ✓ Generated {len(self.predictions):,} predictions")
        
        return self.predictions
    
    def evaluate(self):
        """Evaluate model performance"""
        print("\n📈 Evaluating model performance...")
        
        if self.predictions is None:
            print("   ✗ Please make predictions first!")
            return
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)
        mape = np.mean(np.abs((self.y_test - self.predictions) / self.y_test)) * 100
        
        self.metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        print("\n" + "="*50)
        print("   BASELINE MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"   Mean Squared Error (MSE):  {mse:.6f}")
        print(f"   Root Mean Squared Error:   {rmse:.6f}")
        print(f"   Mean Absolute Error (MAE): {mae:.6f}")
        print(f"   R² Score:                  {r2:.6f}")
        print(f"   MAPE:                      {mape:.2f}%")
        print("="*50)
        
        return self.metrics
    
    def visualize_predictions(self, save_path='reports/figures/baseline_predictions.png'):
        """Visualize actual vs predicted values"""
        print("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(self.y_test, self.predictions, alpha=0.5, s=10)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 
                        'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted Energy Consumption')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = self.y_test - self.predictions
        axes[0, 1].scatter(self.predictions, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Time series comparison (first 500 points)
        n_points = min(500, len(self.y_test))
        axes[1, 0].plot(range(n_points), self.y_test.values[:n_points], 
                       label='Actual', alpha=0.7, linewidth=1.5)
        axes[1, 0].plot(range(n_points), self.predictions[:n_points], 
                       label='Predicted', alpha=0.7, linewidth=1.5)
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Energy Consumption')
        axes[1, 0].set_title(f'Actual vs Predicted (First {n_points} points)')
        axes[1, 0].legend()
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
        print(f"   ✓ Visualization saved to {save_path}")
        plt.show()
    
    def save_model(self, path='models/baseline_lr.pkl'):
        """Save trained model"""
        if self.model is None:
            print("   ✗ No model to save!")
            return
        
        joblib.dump(self.model, path)
        print(f"   ✓ Model saved to {path}")
    
    def load_model(self, path='models/baseline_lr.pkl'):
        """Load trained model"""
        self.model = joblib.load(path)
        print(f"   ✓ Model loaded from {path}")
        return self.model
    
    def predict_future(self, X_future):
        """
        Make predictions on future data
        
        Args:
            X_future: Future feature data
        """
        if self.model is None:
            print("   ✗ Please train or load model first!")
            return
        
        predictions = self.model.predict(X_future)
        return predictions


def main():
    """Main execution function"""
    print("="*60)
    print("SMART ENERGY ANALYSIS - BASELINE MODEL")
    print("="*60)
    
    # Load feature-engineered data
    print("\n📂 Loading feature-engineered data...")
    df = pd.read_csv('data/processed/energy_data_features.csv', 
                     index_col='time', parse_dates=True)
    print(f"   ✓ Loaded {len(df):,} records with {len(df.columns)} features")
    
    # Initialize baseline model
    baseline = BaselineModel()
    
    # Prepare data
    target_col = 'use_HO' if 'use_HO' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    print(f"\n   Target variable: {target_col}")
    
    X_train, X_test, y_train, y_test = baseline.prepare_data(df, target_col)
    
    # Train model
    baseline.train()
    
    # Make predictions
    predictions = baseline.predict()
    
    # Evaluate model
    metrics = baseline.evaluate()
    
    # Visualize results
    baseline.visualize_predictions()
    
    # Save model
    baseline.save_model()
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('reports/results/baseline_metrics.csv', index=False)
    print(f"\n   ✓ Metrics saved to reports/results/baseline_metrics.csv")
    
    print("\n" + "="*60)
    print("✓ BASELINE MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return baseline, metrics


if __name__ == "__main__":
    baseline, metrics = main()
