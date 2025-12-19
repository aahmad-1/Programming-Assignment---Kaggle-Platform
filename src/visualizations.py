import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('./outputs/plots', exist_ok=True)


def plot_regression_results(y_true, y_pred, model_name, save_path):
    """Plot actual vs predicted values and residuals."""
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual G3')
    plt.ylabel('Predicted G3')
    plt.title(f'Actual vs. Predicted - {model_name}')
    plt.grid(True)
    
    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted G3')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {model_name}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_factor_importance(factor_importance, model_name, save_path, top_n=15):
    top_factors = factor_importance.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_factors)), top_factors['importance'])
    plt.yticks(range(len(top_factors)), top_factors['factor'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features - {model_name}')
    plt.gca().invert_yaxis()    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    predictions_path = './outputs/predictions.npy'
    importance_path = './outputs/factor_importance.npy'
    
    if not os.path.exists(predictions_path):
        print(f"ERROR: File not found: {predictions_path}")
        print("Please run train.py first to generate predictions.")
        exit(1)
    
    if not os.path.exists(importance_path):
        print(f"ERROR: File not found: {importance_path}")
        print("Please run train.py first to generate feature importance data.")
        exit(1)
    
    # Load pred and factor importance
    predictions = np.load(predictions_path, allow_pickle=True).item()
    factor_importance = np.load(importance_path, allow_pickle=True).item()
    
    for key, (y_true, y_pred) in predictions.items():
        model_type = "Linear Regression" if "lr" in key else "Random Forest"
        if "with_g1g2" in key:
            g1g2_status = "With G1 & G2"
        elif "without_g1g2" in key:
            g1g2_status = "Without G1 & G2"
        else:
            g1g2_status = "Unknown"
        model_name = f"{model_type} ({g1g2_status})"
        save_path = f"./outputs/plots/{key}.png"
        
        plot_regression_results(y_true, y_pred, model_name, save_path)
    
    for key, importance_df in factor_importance.items():
        if "with_g1g2" in key:
            g1g2_status = "With G1 & G2"
        elif "without_g1g2" in key:
            g1g2_status = "Without G1 & G2"
        else:
            g1g2_status = "Unknown"
        model_name = f"Random Forest ({g1g2_status})"
        save_path = f"./outputs/plots/{key}_importance.png"
        
        plot_factor_importance(importance_df, model_name, save_path)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*60)