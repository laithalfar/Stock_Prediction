"""
evaluate.py
===========
Functions to evaluate trained models on test data and report metrics.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error, explained_variance_score
import os
import sys
import joblib

# Add project root to Python path
sys.path.append(os.path.abspath(".."))

from src.train import train_pipeline, plot_training_history
from config import MODEL_DIR

def calculate_psi(expected, actual, buckets=10):
    # quantile edges from expected (baseline) distribution
    quantiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    # guard against duplicate edges
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    exp_cnt, _ = np.histogram(expected, bins=quantiles)
    act_cnt, _ = np.histogram(actual,  bins=quantiles)

    # convert to proportions with small epsilon
    eps = 1e-6
    exp_p = (exp_cnt + eps) / (exp_cnt.sum() + eps * buckets)
    act_p = (act_cnt + eps) / (act_cnt.sum() + eps * buckets)

    psi = (exp_p - act_p) * np.log(exp_p / act_p)
    return psi.sum()


def evaluate_distribution_shift(X_train, X_test, feature="Close", use_z=False):
    tr = X_train[feature].dropna().to_numpy()
    te = X_test[feature].dropna().to_numpy()
    if use_z:
        mu, sigma = tr.mean(), tr.std(ddof=0) + 1e-8
        tr = (tr - mu) / sigma
        te = (te - mu) / sigma
    psi = calculate_psi(tr, te)
    tag = f"{feature}_z" if use_z else feature
    print(f"PSI for {tag}: {psi:.4f}")
    return psi

def mase(y_true, y_pred, close_series):
    # Naive forecast: shift close_series by 1
    naive_preds = close_series[:-1]  
    naive_true  = close_series[1:]   # align to compare
    
    # MAE of your model
    mae_model = mean_absolute_error(y_true, y_pred)
    
    # MAE of naive baseline
    mae_naive = mean_absolute_error(naive_true, naive_preds)
    
    return mae_model / mae_naive


def smape(y_true, y_pred):
    return 100 * np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def evaluate_model(model, X_test, y_test, close_series):
    """Evaluate the trained model on test data."""
    print("[INFO] Evaluating model on test data...")

    try:
        # Base loss from model
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"[INFO] Test Loss: {test_loss:.6f}")

        # Predictions
        predictions = model.predict(X_test, verbose=0)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        mase_value = mase(y_test, predictions, close_series)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        sMape = smape(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        medae = median_absolute_error(y_test, predictions)
        evs = explained_variance_score(y_test, predictions)

        print(f"[INFO] Additional Metrics:")
        print(f"  MAE:   {mae:.6f}")
        print(f"  MASE:  {mase_value:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  sMAPE:  {sMape:.2f}%")
        print(f"  R²:    {r2:.6f}")
        print(f"  MedAE: {medae:.6f}")
        print(f"  EVS:   {evs:.6f}")

        return {
            'test_loss': test_loss,
            'mae': mae,
            'rmse': rmse,
            'sMape': sMape,
            'r2': r2,
            'medae': medae,
            'evs': evs,
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
        best_model_path = MODEL_DIR / "best_model.keras"
        best_model.save(best_model_path)
        print(f"[INFO] Best model saved to: {best_model_path}")

        return best_model_path

def reconstruct_close_from_returns(y_pred_returns, y_true_returns, close_series):
    """
    Reconstruct predicted Close prices from predicted next-day returns.
    
    Parameters
    ----------
    y_pred_returns : np.ndarray
        Predicted next-day returns, shape (n_samples,)
    y_true_returns : np.ndarray
        Actual next-day returns, shape (n_samples,)
    close_series : np.ndarray or pd.Series
        Actual close prices aligned so that Close[t] corresponds to the base for Return[t+1].
        Must have length n_samples + 1.
    
    Returns
    -------
    pd.DataFrame with:
        - Close_t: base close at time t
        - True_Close_t+1: actual close at t+1
        - Pred_Close_t+1: reconstructed predicted close at t+1
        - True_Return: actual return
        - Pred_Return: predicted return
    """
    # sanity check
    if len(close_series) != len(y_true_returns) + 1:
        raise ValueError("close_series must be one element longer than return arrays")

    # base closes (time t)
    close_t = np.array(close_series[:-1])
    # true next-day closes
    true_close_next = np.array(close_series[1:])

    # reconstruct predictions
    pred_close_next = close_t * (1 + y_pred_returns)

    return pd.DataFrame({
        "Close_t": close_t,
        "True_Close_t+1": true_close_next,
        "Pred_Close_t+1": pred_close_next,
        "True_Return": y_true_returns,
        "Pred_Return": y_pred_returns
    })

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
            results = evaluate_model(train_results["model_list"][f], train_results["X_te_list"][f], train_results["y_te_scaled_list"][f], train_results["close_te_list"][f])

            # inside main(), in the fold loop after evaluate_model()
            y_pred_scaled = results['predictions'].flatten()
            y_test_scaled = train_results["y_te_scaled_list"][f].flatten()

            # 1. inverse transform to returns
           # use the target scaler
            y_scaler = train_results["y_scaler_list"][f]
            y_pred_returns = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true_returns = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

            # robust std from training returns for this fold (recompute from X_tr/y_tr if you kept them unscaled,
            # or pass train returns through the same y_scaler inverse as needed)
            ret_std = np.std(y_true_returns) if np.std(y_true_returns) > 0 else 1e-3
            cap = 3.0 * ret_std  # 3-sigma clamp; tune 2.5–4.0 if needed
            y_pred_returns = np.clip(y_pred_returns, -cap, cap)

            # 2. grab Close prices aligned with this fold's test set
            # assume you saved or can access the original dataframe slices (Close column)
            # For now, say you stored it in train_results["close_te_list"][f]
            close_series = train_results["close_te_list"][f] 

            # 🔍 Debug check for alignment
            print(f"Fold {f+1}: len(y_true_returns)={len(y_true_returns)}, len(close_series)={len(close_series)}")
            print(f"First 5 closes: {close_series[:5]}")
            print(f"First 5 returns (true): {y_true_returns[:5]}")

            # 3. reconstruct closes
            reconstructed = reconstruct_close_from_returns(y_pred_returns, y_true_returns, close_series)

            # 4. compute close-level metrics
            close_mae = mean_absolute_error(reconstructed["True_Close_t+1"], reconstructed["Pred_Close_t+1"])
            close_rmse = np.sqrt(mean_squared_error(reconstructed["True_Close_t+1"], reconstructed["Pred_Close_t+1"]))

            # normalize by average actual Close in this fold
            avg_close = np.mean(train_results["close_te_list"][f])

            close_mae_pct = (close_mae / avg_close) * 100
            close_rmse_pct = (close_rmse / avg_close) * 100 

            print(f"[INFO] Fold {f+1} Close-level Metrics:")
            print(f"  Close MAE:  {close_mae:.4f}")
            print(f"  Close RMSE: {close_rmse:.4f}")
            print(f"  Close MAE percentage:   ({close_mae_pct:.2f}%)")
            print(f"  Close RMSE percentage:  ({close_rmse_pct:.2f}%)")


            # Save metrics
            scores.append(results['rmse'])
            fold_results.append({
                "fold": f + 1 ,
                "mae": results['mae'],
                "rmse": results['rmse'],
                "close_mae": close_mae,            # new
                "close_rmse": close_rmse,  
                "test_loss": results['test_loss'],
                'sMape': results['sMape'],
                'r2': results['r2'],
                'medae': results["medae"],
                'evs': results['evs'],
                'predictions': results['predictions']
            })

            # Collapse samples × timesteps into rows
            X_te_arr = train_results["X_te_list"][f].reshape(-1, train_results["X_te_list"][f].shape[-1])   # shape becomes e.g.(21*10, 16)
            X_tr_arr = train_results["X_tr_list"][f].reshape(-1, train_results["X_tr_list"][f].shape[-1])   # shape becomes e.g.(21*10, 16)


             # 🔽 NEW: check distribution shift for "Close"
            # after making X_te_arr, X_tr_arr
            psi = evaluate_distribution_shift(
                pd.DataFrame(X_tr_arr, columns=train_results["feature_columns"]),
                pd.DataFrame(X_te_arr, columns=train_results["feature_columns"]),
                feature="Close",
            )
            print(f"[INFO] Fold {f+1} PSI for 'Close': {psi:.4f}")

            # Plot training history
            plot_training_history(train_results["history_list"][f], f)

        #Save fold-level results for analysis
        results_df = pd.DataFrame(fold_results)
        results_path = os.path.join(MODEL_DIR, "walk_forward_results.csv")
        results_df.to_csv(results_path, index=False)
        #print(f"[INFO] Fold results saved to: {results_path}")

        # Save best model separately
        best_model_path = calculate_best_model(scores, fold_results, train_results)

        # Load best model to verify
        best_model = load_model(best_model_path)
        print("best model: ")
        best_model.summary()    

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