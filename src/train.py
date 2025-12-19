import os
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_prep import load_and_preprocess

np.random.seed(42)

output_dir = './outputs'
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            for subfile in os.listdir(file_path):
                os.remove(os.path.join(file_path, subfile))
            os.rmdir(file_path)
else:
    os.makedirs(output_dir)

os.makedirs('./outputs/metrics', exist_ok=True)


def train_linear_regression(X_train, X_test, y_train, y_test, model_name):
    """Train Linear Regression model and return metrics."""
    print(f"\n{'='*50}")
    print(f"LINEAR REGRESSION: {model_name}")
    print('='*50)
    
    # Train model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    metrics = {
        'model': f'Linear Regression - {model_name}',
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'rmse_train': math.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test': math.sqrt(mean_squared_error(y_test, y_pred_test))
    }
    
    # Cross val
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='r2')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std() * 2
    
    print(f"Training R²: {metrics['r2_train']:.4f}")
    print(f"Testing R²: {metrics['r2_test']:.4f}")
    print(f"Training MAE: {metrics['mae_train']:.4f}")
    print(f"Testing MAE: {metrics['mae_test']:.4f}")
    print(f"Training RMSE: {metrics['rmse_train']:.4f}")
    print(f"Testing RMSE: {metrics['rmse_test']:.4f}")
    print(f"Mean CV R²: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    return lr_model, y_pred_test, metrics


def train_random_forest(X_train, X_test, y_train, y_test, model_name, 
                        n_estimators=100, max_depth=10):
    
    print(f"\n{'='*50}")
    print(f"RANDOM FOREST: {model_name}")
    print('='*50)
    
    # Train model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    metrics = {
        'model': f'Random Forest - {model_name}',
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'rmse_train': math.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test': math.sqrt(mean_squared_error(y_test, y_pred_test))
    }
    
    # Cross val
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std() * 2
    
    # Factor importance
    factor_importance = pd.DataFrame({
        'factor': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Training R²: {metrics['r2_train']:.4f}")
    print(f"Testing R²: {metrics['r2_test']:.4f}")
    print(f"Training MAE: {metrics['mae_train']:.4f}")
    print(f"Testing MAE: {metrics['mae_test']:.4f}")
    print(f"Training RMSE: {metrics['rmse_train']:.4f}")
    print(f"Testing RMSE: {metrics['rmse_test']:.4f}")
    print(f"Mean CV R²: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    print(f"\nTop 10 most important factors:")
    print(factor_importance.head(10).to_string(index=False))
    
    return rf_model, y_pred_test, factor_importance, metrics


def save_results(all_metrics):

    output_file = './outputs/metrics/results.txt'
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("STUDENT PERFORMANCE PREDICTION - MODEL COMPARISON\n")
        f.write("="*60 + "\n\n")
        
        for metrics in all_metrics:
            f.write(f"\nModel: {metrics['model']}\n")
            f.write("-"*60 + "\n")
            f.write(f"Training R²:    {metrics['r2_train']:.4f}\n")
            f.write(f"Testing R²:     {metrics['r2_test']:.4f}\n")
            f.write(f"Training MAE:   {metrics['mae_train']:.4f}\n")
            f.write(f"Testing MAE:    {metrics['mae_test']:.4f}\n")
            f.write(f"Training RMSE:  {metrics['rmse_train']:.4f}\n")
            f.write(f"Testing RMSE:   {metrics['rmse_test']:.4f}\n")
            f.write(f"CV R² (mean):   {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    all_metrics = []
    all_predictions = {}
    all_factor_importance = {}
    
    print("\n" + "="*60)
    print("TRAINING MODELS WITH G1 AND G2")
    print("="*60)
    
    X_train, X_test, y_train, y_test, factor_names = load_and_preprocess(include_g1g2=True)
    
    # Train Linear Regression with G1 and G2
    lr_model_with, lr_pred_with, lr_metrics_with = train_linear_regression(
        X_train, X_test, y_train, y_test, "With G1 & G2"
    )
    all_metrics.append(lr_metrics_with)
    all_predictions['lr_with_g1g2'] = (y_test, lr_pred_with)
    
    # Train Random Forest with G1 and G2
    rf_model_with, rf_pred_with, rf_importance_with, rf_metrics_with = train_random_forest(
        X_train, X_test, y_train, y_test, "With G1 & G2"
    )
    all_metrics.append(rf_metrics_with)
    all_predictions['rf_with_g1g2'] = (y_test, rf_pred_with)
    all_factor_importance['rf_with_g1g2'] = rf_importance_with
    
    print("\n" + "="*60)
    print("TRAINING MODELS WITHOUT G1 AND G2")
    print("="*60)
    
    X_train, X_test, y_train, y_test, factor_names = load_and_preprocess(include_g1g2=False)
    
    # Train Linear Regression without G1 and G2
    lr_model_without, lr_pred_without, lr_metrics_without = train_linear_regression(
        X_train, X_test, y_train, y_test, "Without G1 & G2"
    )
    all_metrics.append(lr_metrics_without)
    all_predictions['lr_without_g1g2'] = (y_test, lr_pred_without)
    
    # Train Random Forest without G1 and G2
    rf_model_without, rf_pred_without, rf_importance_without, rf_metrics_without = train_random_forest(
        X_train, X_test, y_train, y_test, "Without G1 & G2"
    )
    all_metrics.append(rf_metrics_without)
    all_predictions['rf_without_g1g2'] = (y_test, rf_pred_without)
    all_factor_importance['rf_without_g1g2'] = rf_importance_without
    
    save_results(all_metrics)
    
    # Save results for visualizations.py
    np.save('./outputs/predictions.npy', all_predictions, allow_pickle=True)
    np.save('./outputs/factor_importance.npy', all_factor_importance, allow_pickle=True)
    
    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)