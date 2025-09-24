import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys
from train import walk_forward_validation, train_pipeline
from config import MODEL_TYPE, MODEL_SAVE_PATH


def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) between two distributions.
    expected: baseline (train set column)
    actual: comparison (test/val set column)
    """
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_percents = np.histogram(
        np.percentile(expected, breakpoints), bins=buckets
    )[0] / len(expected)
    actual_percents = np.histogram(
        np.percentile(actual, breakpoints), bins=buckets
    )[0] / len(actual)
    
    psi_values = (expected_percents - actual_percents) * np.log(
        (expected_percents + 1e-6) / (actual_percents + 1e-6)
    )
    return np.sum(psi_values)

# Example usage inside evaluation
def evaluate_distribution_shift(X_train, X_test, feature="Close"):
    train_col = X_train[feature].dropna().values
    test_col = X_test[feature].dropna().values
    psi = calculate_psi(train_col, test_col)
    print(f"PSI for {feature}: {psi:.4f}")
    return psi

"""
evaluate.py
===========
Functions to evaluate trained models on test data and report metrics.
"""


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data."""
    print("[INFO] Evaluating model on test data...")

    try:
        # Base loss from model
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"[INFO] Test Loss: {test_loss:.6f}")

        # Predictions
        predictions = model.predict(X_test, verbose=0)

        # Metrics
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        rmse = np.sqrt(np.mean((predictions.flatten() - y_test) ** 2))

        print(f"[INFO] Additional Metrics:")
        print(f"  MAE: {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")

        return {
            'test_loss': test_loss,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions
        }

    except Exception as e:
        print(f"[ERROR] Model evaluation failed: {e}")
        raise


if __name__ == "__main__":
    print("[INFO] This module is intended to be imported, not run directly.")

#main function
def main():
    try:
        
        model, history, scores, fold_results, X_te, y_te = train_pipeline()

        # Assuming last fold's test set for final evaluation
        for f in fold_results:
        
            # Evaluate
            results = evaluate_model(model, X_te, y_te)
            scores.append(results['rmse'])
            fold_results.append({
                "fold": fold_results+1,
                "mae": results['mae'],
                "rmse": results['rmse'],
                "test_loss": results['test_loss']
            })

        # 3. Aggregate results
        print("="*60)
        print(f"Walk-Forward Validation Complete. Avg RMSE: {np.mean(scores):.4f}")
        print("="*60)

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        raise