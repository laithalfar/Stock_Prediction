"""
evaluate.py
===========
Functions to evaluate trained models on test data and report metrics.
"""

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,  median_absolute_error, explained_variance_score
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(".."))

from src.train import train_pipeline, plot_training_history
from config import MODEL_DIR, MODEL_TYPE

# Calculate psi to see if there is distribution shift
def calculate_psi(expected, actual, buckets=10):


    """
    Calculate the PSI (Population Stability Index) metric for a given pair of expected and actual values.

    The PSI is a measure of the distribution shift between the expected and actual values.

    Parameters:
    expected (array-like): The expected values.
    actual (array-like): The actual values.
    buckets (int): The number of quantile bins to use. Defaults to 10.

    Returns:
    psi (float): The PSI metric value.

    """


    quantiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    # guard against duplicate edges
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    # bin counts
    exp_cnt, _ = np.histogram(expected, bins=quantiles)
    act_cnt, _ = np.histogram(actual,  bins=quantiles)

    # convert to proportions with small epsilon
    eps = 1e-6
    exp_p = (exp_cnt + eps) / (exp_cnt.sum() + eps * buckets)
    act_p = (act_cnt + eps) / (act_cnt.sum() + eps * buckets)

    psi = (exp_p - act_p) * np.log(exp_p / act_p)
    return psi.sum()

# Evaluate distribution shift for a specific feature
def evaluate_distribution_shift(X_train, X_test, feature="Close", use_z=False):

    """
    Evaluate the distribution shift of a feature between the training and test data.
    
    Parameters:
    X_train (pd.DataFrame): The training data.
    X_test (pd.DataFrame): The test data.
    feature (str): The feature to evaluate the distribution shift for. Defaults to "Close".
    use_z (bool): Whether to normalize the feature values using Z-score normalization. Defaults to False.
    
    Returns:
    psi (float): The distribution shift of the feature between the training and test data.
    """

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

    """
    Calculate the Mean Absolute Error (MAE) of the model and the naive baseline, then return the ratio of the two MAEs.

    Parameters:
    y_true (array-like): The true values.
    y_pred (array-like): The predicted values.
    close_series (array-like): The close series to use for the naive baseline.

    Returns:
    float: The ratio of the MAE of the model to the MAE of the naive baseline.
    """

    naive_preds = close_series[:-1]  
    naive_true  = close_series[1:]   # align to compare
    
    # MAE of your model
    mae_model = mean_absolute_error(y_true, y_pred)
    
    # MAE of naive baseline
    mae_naive = mean_absolute_error(naive_true, naive_preds)
    
    return mae_model / mae_naive


def smape(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) of the model.

    Parameters:
    y_true (array-like): The true values.
    y_pred (array-like): The predicted values.

    Returns:
    float: The SMAPE metric value.

    """
    return 100 * np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def evaluate_model(model, X_test, y_test, close_series):
    
    """
    Evaluate a model on the test data and return the results.

    Parameters:
    model (keras.Model): The model to evaluate.
    X_test (array-like): The test data.
    y_test (array-like): The true values.
    close_series (array-like): The close series to use for the naive baseline.

    Returns:
    dict: A dictionary containing the evaluation results.
    """
    print("[INFO] Evaluating model on test data...")

    try:
        # Base loss from model
        results = model.evaluate(X_test, y_test, verbose=0)
        #Handle both scalar and list return types safely
        if isinstance(results, list):
            test_loss = results[0]
            extra_metrics = results[1:]
        else:
            test_loss = results
            extra_metrics = []

        # --- Safe logging ---
        print(f"[INFO] Test Loss: {test_loss:.6f}")

        if extra_metrics:
            for i, val in enumerate(extra_metrics, start=1):
                try:
                    print(f"[INFO] Metric {i}: {val:.6f}")
                except TypeError:
                    print(f"[INFO] Metric {i}: {val}")  # fallback if not numeric

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
                
    """
    Calculate the best model based on the lowest RMSE.
    
    Parameters
    ----------
    scores : list
        List of RMSE values for each fold.
    fold_results : list
        List of dictionaries containing the results for each fold.
    train_results : dict
        Dictionary containing the model list and other results.
    
    Returns
    -------
    str
        Path to the best model saved for deployment.
    """
    best_idx = np.argmin(scores) # index of best fold scores
    best_model = train_results["model_list"][best_idx] # corresponding model
    best_metrics = fold_results[best_idx] # corresponding metrics

    print("="*60)
    print(f"Best model is from Fold {best_idx+1} with RMSE={best_metrics['rmse']:.4f}, "
              f"MAE={best_metrics['mae']:.4f}, Loss={best_metrics['test_loss']:.4f}")
    print("="*60)

    # Optional: save separately for deployment
    best_model_path = MODEL_DIR / f"results/{MODEL_TYPE}_results/best_model.keras"
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



def check_prediction_volatility(y_true_returns, y_pred_returns, sigma_mult=3.0):
    """
    Diagnose whether your model's predicted returns need clipping.
    
    Parameters
    ----------
    y_true_returns : np.ndarray
        True observed returns.
    y_pred_returns : np.ndarray
        Model-predicted returns.
    sigma_mult : float
        The sigma multiplier for the limit (default 3.0 → ±3σ window).
    """
    # Compute standard deviation of actual returns
    ret_std = np.std(y_true_returns)
    if ret_std <= 0:
        ret_std = 1e-3  # avoid div by zero
    
    cap = sigma_mult * ret_std
    extreme_mask = np.abs(y_pred_returns) > cap
    num_extreme = np.sum(extreme_mask)
    pct_extreme = 100 * num_extreme / len(y_pred_returns)

    print("="*60)
    print(f"Return volatility diagnostic:")
    print(f"  σ of true returns:     {ret_std:.6f}")
    print(f"  ±{sigma_mult}σ cap:    ±{cap:.6f}")
    print(f"  # of predictions > cap: {num_extreme} ({pct_extreme:.2f}%)")
    print("="*60)

    # Return indices or boolean mask of extreme predictions if you want to inspect them
    return extreme_mask

# Main function
def main():
    """
    Main function to evaluate a trained model on test data using walk-forward validation.

    This function takes in a trained model and evaluates it on test data using walk-forward validation.
    It returns a dictionary containing the test metrics and the predicted Close prices.

    The function first reconstructs the predicted Close prices from the predicted returns.
    It then computes the close-level metrics (MAE, RMSE) and normalizes them by the average actual Close in each fold.
    Finally, it saves the fold-level results for analysis and saves the best model separately.

    Parameters
    ----------
    model : keras.Model
        Trained model to evaluate.
    X_te_list : list
        List of test feature arrays.
    y_te_scaled_list : list
        List of test target arrays (scaled).
    close_te_list : list
        List of test Close prices aligned with the test sets.
    feature_columns : list
        List of feature column names.

    Returns
    -------
    dict
        Dictionary containing the test metrics and the predicted Close prices.
    """
    try:
         
        #results for training pipeline
        train_results = train_pipeline()


        # Initialize lists
        scores, fold_results = [], []

        # Loop over folds
        for f in range(len(train_results["X_te_list"])):
        
            # Evaluate
            results = evaluate_model(train_results["model_list"][f], train_results["X_te_list"][f], train_results["y_te_scaled_list"][f], train_results["close_te_list"][f])

            # Inside main(), in the fold loop after evaluate_model()
            y_pred_scaled = results['predictions'].flatten()
            y_test_scaled = train_results["y_te_scaled_list"][f].flatten()

            # 1. inverse transform to returns
            # Use the target scaler
            y_scaler = train_results["y_scaler_list"][f]
            y_pred_returns = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true_returns = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

            # check if predictions are too volatile and must be clipped
            check_prediction_volatility(y_true_returns, y_pred_returns)


            # 2. grab Close prices aligned with this fold's test set
            # Assume you saved or can access the original dataframe slices (Close column)
            # For now, say you stored it in train_results["close_te_list"][f]
            close_series = train_results["close_te_list"][f] 

            # 🔍 Debug check for alignment
            print(f"Fold {f+1}: len(y_true_returns)={len(y_true_returns)}, len(close_series)={len(close_series)}")
            print(f"First 5 closes: {close_series[:5]}")
            print(f"First 5 returns (true): {y_true_returns[:5]}")

            # 3. Reconstruct closes from returns
            reconstructed = reconstruct_close_from_returns(y_pred_returns, y_true_returns, close_series)

            # 4. Compute close-level metrics
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
                use_z=True
            )
            print(f"[INFO] Fold {f+1} PSI for 'Close': {psi:.4f}")

            # Plot training history
            plot_training_history(train_results["history_list"][f], f)

        #Save fold-level results for analysis
        results_df = pd.DataFrame(fold_results)
        results_path = os.path.join(MODEL_DIR, f"results/{MODEL_TYPE}_results/walk_forward_results.csv")
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