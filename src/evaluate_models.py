"""
Module 6: Model Evaluation & Comparison
Compares the performance of baseline (Linear Regression) and LSTM models.
"""
import sys
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results():
    """Load prediction results from both models."""
    with open('results/baseline_predictions.pkl', 'rb') as f:
        baseline = pickle.load(f)
    with open('results/lstm_predictions.pkl', 'rb') as f:
        lstm = pickle.load(f)
    return baseline, lstm


def compare_metrics(baseline, lstm):
    """Compare and display metrics from both models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON - Test Set Performance")
    print("=" * 60)
    
    metrics = ['mae', 'rmse', 'r2']
    metric_names = ['MAE', 'RMSE', 'R²']
    
    print(f"\n{'Metric':<12} {'Baseline (LR)':<18} {'LSTM':<18} {'Improvement':<15}")
    print("-" * 60)
    
    improvements = []
    for metric, name in zip(metrics, metric_names):
        baseline_val = baseline['metrics'][metric]
        lstm_val = lstm['metrics'][metric]
        
        if metric == 'r2':
            # For R2, higher is better
            if np.isnan(lstm_val):
                improvement = "N/A"
            else:
                improvement = f"{((lstm_val - baseline_val) / abs(baseline_val)) * 100:+.1f}%"
        else:
            # For MAE/RMSE, lower is better
            improvement = f"{((baseline_val - lstm_val) / baseline_val) * 100:+.1f}%"
        
        improvements.append(improvement)
        print(f"{name:<12} {baseline_val:<18.6f} {lstm_val:<18.6f} {improvement:<15}")
    
    print("-" * 60)
    
    # Determine winner
    if lstm['metrics']['mae'] < baseline['metrics']['mae']:
        print("\n✓ LSTM model performs BETTER on MAE metric.")
    else:
        print("\n✓ Baseline model performs BETTER on MAE metric.")
    
    return improvements


def create_comparison_plots(baseline, lstm):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics comparison bar chart
    metrics = ['MAE', 'RMSE']
    baseline_vals = [baseline['metrics']['mae'], baseline['metrics']['rmse']]
    lstm_vals = [lstm['metrics']['mae'], lstm['metrics']['rmse']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, baseline_vals, width, label='Baseline (LR)', color='steelblue')
    bars2 = axes[0].bar(x + width/2, lstm_vals, width, label='LSTM', color='coral')
    
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Value (Lower is Better)')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # R2 comparison (if valid)
    r2_baseline = baseline['metrics']['r2']
    r2_lstm = lstm['metrics']['r2']
    
    if not np.isnan(r2_lstm):
        r2_vals = [r2_baseline, r2_lstm]
        colors = ['steelblue', 'coral']
        bars = axes[1].bar(['Baseline (LR)', 'LSTM'], r2_vals, color=colors)
        axes[1].set_ylabel('R² Score (Higher is Better)')
        axes[1].set_title('R² Score Comparison')
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'R² not available\n(insufficient test data)',
                    ha='center', va='center', fontsize=12,
                    transform=axes[1].transAxes)
        axes[1].set_title('R² Score Comparison')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150)
    plt.close()
    print("\nComparison chart saved to results/model_comparison.png")


def main():
    """Run the evaluation and comparison."""
    print("Loading model results...")
    baseline, lstm = load_results()
    
    print(f"\nBaseline predictions: {len(baseline['y_test'])} samples")
    print(f"LSTM predictions: {len(lstm['y_test'])} samples")
    
    # Compare metrics
    improvements = compare_metrics(baseline, lstm)
    
    # Create visualizations
    create_comparison_plots(baseline, lstm)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
