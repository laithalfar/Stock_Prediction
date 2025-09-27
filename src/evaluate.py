"""
evaluate.py
===========
Functions to evaluate trained models on test data and report metrics.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(".."))

from src.train import train_pipeline, plot_training_history
from config import MODEL_DIR

# Function to calculate PSI
def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) between two distributions.
    expected: baseline (train set column)
    actual: comparison (test/val set column)
    """

    #create array of evenly spaced numbers between 0 and 100
    breakpoints = np.linspace(0, 100, buckets + 1)

    #expected histogram structure 
    expected_percents = np.histogram(
        np.percentile(expected, breakpoints), bins=buckets
    )[0] / len(expected)

    #actual histogram
    actual_percents = np.histogram(
        np.percentile(actual, breakpoints), bins=buckets
    )[0] / len(actual)
    
    psi_values = (expected_percents - actual_percents) * np.log(
        (expected_percents + 1e-6) / (actual_percents + 1e-6)
    )
    return np.sum(psi_values)


# Example usage inside evaluation
def evaluate_distribution_shift(X_train, X_test, feature="Close"):
    #drop NaNs in each set
    train_col = X_train[feature].dropna().values
    test_col = X_test[feature].dropna().values

    # Calculate PSI and print
    psi = calculate_psi(train_col, test_col)
    print(f"PSI for {feature}: {psi:.4f}")
    return psi



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
        mae = mean_absolute_error(y_test, predictions.flatten())
        rmse = np.sqrt(mean_squared_error(y_test, predictions.flatten()))

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

# Function to pick the best model
def calculate_best_model(scores, fold_results, train_results):
     # 🔽 NEW: pick the best fold by lowest RMSE
        best_idx = np.argmin(scores)
        best_model = train_results["model_list"][best_idx]
        best_metrics = fold_results[best_idx]

        print("="*60)
        print(f"Best model is from Fold {best_idx+1} with RMSE={best_metrics['rmse']:.4f}, "
              f"MAE={best_metrics['mae']:.4f}, Loss={best_metrics['test_loss']:.4f}")
        print("="*60)

        # Optional: save separately for deployment
        best_model_path = os.path.join(MODEL_DIR, "best_model.keras")
        best_model.save(best_model_path)
        print(f"[INFO] Best model saved to: {best_model_path}")

        return best_model_path


# Main function
def main():
    try:
        
        #results for training pipeline
        train_results = train_pipeline()



        # Initialize lists
        scores, fold_results = [], []

        # Loop over folds
        for f in range(len(train_results["X_te_list"])):
        
            # Evaluate
            results = evaluate_model(train_results["model_list"][f], train_results["X_te_list"][f], train_results["y_te_list"][f])

            # Save metrics
            scores.append(results['rmse'])
            fold_results.append({
                "fold": f + 1 ,
                "mae": results['mae'],
                "rmse": results['rmse'],
                "test_loss": results['test_loss']
            })

            # Collapse samples × timesteps into rows
            X_te_arr = train_results["X_te_list"][f].reshape(-1, train_results["X_te_list"][f].shape[-1])   # shape becomes e.g.(21*10, 16)
            X_tr_arr = train_results["X_te_list"][f].reshape(-1, train_results["X_te_list"][f].shape[-1])   # shape becomes e.g.(21*10, 16)


             # 🔽 NEW: check distribution shift for "Close"
            # Here we compare the training vs test distribution in this fold
            psi = evaluate_distribution_shift(
            pd.DataFrame(X_te_arr, columns=train_results["feature_columns"]), 
            pd.DataFrame(X_tr_arr, columns=train_results["feature_columns"]),
            feature="Close"
         )
            print(f"[INFO] Fold {f+1} PSI for 'Close': {psi:.4f}")

            # Plot training history
            plot_training_history(train_results["history_list"][f])

        #Save fold-level results for analysis
        results_df = pd.DataFrame(fold_results)
        results_path = results_path = os.path.join(MODEL_DIR, "walk_forward_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"[INFO] Fold results saved to: {results_path}")

        # Save best model separately
        best_model_path = calculate_best_model(scores, fold_results, train_results)

        # Load best model to verify
        best_model = load_model(best_model_path)
        print("best model: ", best_model.summary())

       

        # 3. Aggregate results
        print("="*60)
        print(f"Walk-Forward Validation Complete. Avg RMSE: {np.mean(scores):.4f}")
        print("="*60)

        return fold_results

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    # Execute evaluation pipeline
    main()  