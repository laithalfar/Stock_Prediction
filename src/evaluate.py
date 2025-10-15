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
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.abspath(".."))

from src.train import train_pipeline, plot_training_history
from config import MODEL_DIR, MODEL_TYPE, PLOT_ACTUAL_PREDICTED_PATH

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
    #y_true = y_true.to_numpy().flatten()
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
    print(f"[INFO] Test data shape: {X_test.shape}")

    try:
       
        # Predictions
        predictions = model.predict(X_test, verbose=0).flatten()
        print(f"[INFO] Predictions shape: {predictions.shape}")

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        mase_value = mase(y_test, predictions, close_series)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        sMape = smape(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        medae = median_absolute_error(y_test, predictions)
        evs = explained_variance_score(y_test, predictions)

        errors = predictions - y_test

        bias = np.mean(errors)
        error_std = np.std(errors)

        print(f"Bias (Mean Error): {bias:.4f}")
        print(f"Error Std (Precision Proxy): {error_std:.4f}")

        print(f"[INFO] Additional Metrics:")
        print(f"  MAE:   {mae:.6f}")
        print(f"  MASE:  {mase_value:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  sMAPE:  {sMape:.2f}%")
        print(f"  R²:    {r2:.6f}")
        print(f"  MedAE: {medae:.6f}")
        print(f"  EVS:   {evs:.6f}")

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

        return {
            'test_loss': test_loss,
            'mae': mae,
            'rmse': rmse,
            'sMape': sMape,
            'r2': r2,
            'medae': medae,
            'evs': evs,
            'bias': bias,
            'error_std': error_std,
            'predictions': predictions,
        }

    except Exception as e:
        print(f"[ERROR] Model evaluation failed: {e}")
        raise

# Function to pick the best model
def calculate_best_model(scores, fold_results, train_results):
     # 🔽 NEW: pick the best fold by scores metrics
                
    """
    Calculate the best model based on the scores metrics.
    
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

    # Save best model separately for deployment
    best_model_dir = MODEL_DIR / f"results/{MODEL_TYPE}_results"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_model_dir / "best_model.keras"
    best_model.save(best_model_path)
    print(f"[INFO] Best model saved to: {best_model_path}")

    return best_model_path


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

def model_exists(filename, model_name):
    """
    Check if a model has been recorded in the history file.

    Parameters
    ----------
    filename : str
        Path to the history file.
    model_name : str
        Name of the model to check.

    Returns
    -------
    bool
        True if the model has been recorded, False otherwise.
    """
    if not os.path.exists(filename):
        return False  # file doesn't exist yet → model not recorded

    df = pd.read_csv(filename)
    return model_name in df["model_name"].values

def best_of_3_models():

    # determine path results
    """
    Determine which of the three models (LSTM, RNN, CNN-GRU) performed best
    based on averaged results from all folds.

    Returns
    -------
    pd.Series
        A row from the results DataFrame containing the best model's metrics.
    """
    file = os.path.join(MODEL_DIR, f"results/averaged_results.csv")

    # Read file
    df = pd.read_csv(file)

    # determine which metrics are better with lower values and which are better with higher
    lower_is_better = ["mae", "rmse", "close_rmse", "close_mae",  "test_loss", "sMape", "medae", "psi", "error_std", "bias"]
    higher_is_better = ["r2", "evs"]

    # Rank metrics
    for col in lower_is_better:
        if col in df.columns:
            df[f"{col}_rank"] = df[col].rank(ascending=True)

    for col in higher_is_better:
        if col in df.columns:
            df[f"{col}_rank"] = df[col].rank(ascending=False)

    # Compute overall average rank
    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    df["avg_rank"] = df[rank_cols].mean(axis=1)

   # Pick best overall model
    best_row = df.loc[df["avg_rank"].idxmin()]

    print("===========================================")
    print(f"✅ Best overall model: {best_row['model_name']}")
    print("===========================================")

    return best_row


def plot_actual_predicted( actual, predicted, title="actual vs predicted next day close price"):
    """
    Plot actual and predicted next-day Close prices.

    Parameters
    ----------
    close_t : np.ndarray
        Actual Close prices at time t.
    true_close_next : np.ndarray
        Actual Close prices at time t+1.
    pred_close_next : np.ndarray
        Predicted Close prices at time t+1.
    title : str
        Title for the plot (default: "").

    Returns
    -------
    None
    """
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="True Next Day")
    plt.plot(predicted, label="Predicted Next Day")
    plt.title(title)
    plt.legend()
    

    plot_path = PLOT_ACTUAL_PREDICTED_PATH
    plt.savefig(plot_path)
    plt.show()
    print(f"[INFO] Training plots saved to: {plot_path}")
      


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

        actual = []
        predicted = []

        print(f"[INFO] X_train shape: {train_results["X_train"][0].shape}, y_train shape: {train_results["y_train"][0].shape}")
        print(f"[INFO] X_test shape: {train_results["X_test"][0].shape}, y_test shape: {train_results["y_test"][0].shape}")

        # Loop over folds
        for f in train_results["folds"]:
            
            print(f"\n{'='*60}")
            print(f"Evaluating Fold {f+1}")
            print(f"{'='*60}")
            
            # Get fold-specific data
            X_test_fold = train_results["X_test"][f]
            y_test_fold = train_results["y_test"][f]
            X_scaler_fold = train_results["X_scaler_list"]
            y_scaler_fold = train_results["y_scaler_list"]
            close_series_fold = train_results["close_te_list"][f]
            model_fold = train_results["model_list"][f]
            
            # Get Close column index (case-sensitive)
            try:
                close_col_idx = train_results["feature_columns_X"].index("Close")
            except ValueError:
                print("[ERROR] Close column not found in feature columns")
                raise
                
            
            # Get dimensions
            n_samples, T, n_features = X_test_fold.shape
            
            # Invert scaling back to raw feature space to get actual Close prices
            X2 = X_test_fold.reshape(-1, n_features)
            X2_raw = X_scaler_fold.inverse_transform(X2)
            X_test_raw = X2_raw.reshape(n_samples, T, n_features)
            
            # Extract base Close prices (last timestep of each sample)
            close_t = X_test_raw[:, -1, close_col_idx]
            
            # Evaluate model (predicts scaled returns)
            results = evaluate_model(model_fold, X_test_fold, y_test_fold, close_series_fold)
            

            # Get predicted returns (scaled) and inverse transform to actual returns
            y_pred_scaled = results['predictions'].flatten()
            y_test_scaled = y_test_fold
            
            # Inverse transform to get actual returns
            y_pred_returns = y_scaler_fold.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true_returns = y_scaler_fold.inverse_transform(y_test_scaled.to_numpy().reshape(-1, 1)).flatten()
            
            # Check prediction volatility (optional diagnostic)
            # check_prediction_volatility(y_true_returns, y_pred_returns)
            
            # Reconstruct actual Close prices from returns
            # close_t = base prices, returns = percentage changes
            pred_close_next = close_t * (1 + y_pred_returns)
            
            # Get true next-day closes from the close_series (skip first T timesteps used for sequence)
            # close_series_fold includes the full Close column for this test window
            true_close_next = close_series_fold.values[T:]  # Skip first T timesteps
            
            # Ensure alignment
            min_len = min(len(pred_close_next), len(true_close_next), len(close_t))
            pred_close_next = pred_close_next[:min_len]
            true_close_next = true_close_next[:min_len]
            close_t = close_t[:min_len]
            
            # Store in dict for consistency
            reconstructed = {
                "Close_t": close_t,
                "True_Close_t+1": true_close_next,
                "Pred_Close_t+1": pred_close_next
            }
            
            # Compute close-level metrics
            close_mae = mean_absolute_error(reconstructed["True_Close_t+1"], reconstructed["Pred_Close_t+1"])
            close_rmse = np.sqrt(mean_squared_error(reconstructed["True_Close_t+1"], reconstructed["Pred_Close_t+1"]))

            # Normalize by average actual Close in this fold
            avg_close = np.mean(close_series_fold)

            close_mae_pct = (close_mae / avg_close) * 100
            close_rmse_pct = (close_rmse / avg_close) * 100 

            print(f"[INFO] Fold {f+1} Close-level Metrics:")
            print(f"  Close MAE:  {close_mae:.4f}")
            print(f"  Close RMSE: {close_rmse:.4f}")
            print(f"  Close MAE percentage:   ({close_mae_pct:.2f}%)")
            print(f"  Close RMSE percentage:  ({close_rmse_pct:.2f}%)")

            # Collapse samples × timesteps into rows
            X_te_arr = train_results["X_test"][f].reshape(-1, train_results["X_test"][f].shape[-1])   # shape becomes e.g.(21*10, 16)
            X_tr_arr = train_results["X_train"][f].reshape(-1, train_results["X_train"][f].shape[-1])   # shape becomes e.g.(21*10, 16)


            # Check distribution shift for "Close"
            try:
                psi = evaluate_distribution_shift(
                    pd.DataFrame(X_tr_arr, columns=train_results["feature_columns_X"]),
                    pd.DataFrame(X_te_arr, columns=train_results["feature_columns_X"]),
                    feature="Close",
                    use_z=True
                )
            except (KeyError, ValueError) as e:
                print(f"[WARNING] Could not calculate PSI: {e}")
                psi = 0.0

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
                'predictions': results['predictions'],
                'psi': psi,
                'bias': results['bias'],
                'error_std': results['error_std']
            })

            # Plot training history
            plot_training_history(train_results["history_list"][f], f)

            # Accumulate predictions across folds for final plot
            if len(actual) == 0:
                actual = reconstructed["True_Close_t+1"]
                predicted = reconstructed["Pred_Close_t+1"]
            else:
                actual = np.concatenate((actual, reconstructed["True_Close_t+1"]))
                predicted = np.concatenate((predicted, reconstructed["Pred_Close_t+1"]))


        # Plot actual vs predicted
        plot_actual_predicted(actual, predicted)    

        #Save fold-level results for analysis
        results_df = pd.DataFrame(fold_results)
        results_dir = MODEL_DIR / f"results/{MODEL_TYPE}_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "walk_forward_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"[INFO] Saved fold results to: {results_path}")

        # Compute means (and optional standard deviations)
        mean_results = results_df.mean(numeric_only=True)

        # Add identifying info
        mean_results["model_name"] = MODEL_TYPE
        mean_results["folds"] = len(fold_results)

        # Combine into a single-row DataFrame
        mean_df = pd.DataFrame([mean_results])
        mean_df.drop(columns=["fold"], inplace=True)

        results_dir_avg = MODEL_DIR / "results"
        results_dir_avg.mkdir(parents=True, exist_ok=True)
        results_path_avg = results_dir_avg / "averaged_results.csv"
        
        if not results_path_avg.exists():
            mean_df.to_csv(results_path_avg, index=False)
            print(f"[INFO] Created averaged results file: {results_path_avg}")
        elif not model_exists(str(results_path_avg), MODEL_TYPE):
            mean_df.to_csv(results_path_avg, mode="a", index=False, header=False)
            print(f"[INFO] Appended {MODEL_TYPE} results to: {results_path_avg}")

        # Save best model separately
        best_model_path = calculate_best_model(scores, fold_results, train_results)

        # Load best model to verify
        best_model = load_model(best_model_path)
        print("best model: ")
        best_model.summary()

        # Check if all three models have been evaluated
        r = 0
        for m in ["cnn_gru", "lstm", "rnn"]:
            if model_exists(str(results_path_avg), m):
                r = r+1
        
        if r == 3:
            print("\n[INFO] All three models evaluated. Comparing...")
            best_of_3_models()   

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