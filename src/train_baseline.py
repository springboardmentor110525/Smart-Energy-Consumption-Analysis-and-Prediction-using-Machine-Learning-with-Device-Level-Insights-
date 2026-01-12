"""
Module 4: Baseline Model - Linear Regression
Trains a Linear Regression model on the processed energy data.
"""
import sys
import os
import pickle
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression
from src.utils import load_data, prepare_data, evaluate_model


def train_baseline():
    """Train and evaluate the Linear Regression baseline model."""
    print("Loading data...")
    train_df, test_df = load_data()
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    
    # Prepare features and target
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(train_df, test_df)
    
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Train Linear Regression model
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate
    print("\n--- Training Set ---")
    train_metrics = evaluate_model(y_train, y_train_pred, "Linear Regression (Train)")
    
    print("\n--- Test Set ---")
    test_metrics = evaluate_model(y_test, y_test_pred, "Linear Regression (Test)")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/linear_regression.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to models/linear_regression.pkl")
    
    # Save predictions for later comparison
    os.makedirs('results', exist_ok=True)
    results = {
        'y_test': y_test,
        'y_pred': y_test_pred,
        'metrics': test_metrics,
        'index': test_df.index
    }
    with open('results/baseline_predictions.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(y_test)), y_test, label='Actual', alpha=0.7)
    plt.plot(range(len(y_test)), y_test_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Energy Consumption (Normalized)')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/baseline_predictions.png', dpi=150)
    plt.close()
    print("Visualization saved to results/baseline_predictions.png")
    
    return model, test_metrics


if __name__ == "__main__":
    train_baseline()
