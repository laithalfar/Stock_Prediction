"""
predict.py
==========
Use the trained best model to make predictions on new/unseen data.
"""

import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(".."))

from config import MODEL_DIR, SCALER_X_PATH, SCALER_Y_PATH
from src.data_preprocessing import feature_engineering   # assuming this exists
from src.evaluate import reconstruct_close_from_returns, check_prediction_volatility

def load_scaler(path):
    
    """
    Load a saved StandardScaler from a file.

    Parameters
    ----------
    path : str
        Path to the saved StandardScaler file.

    Returns
    -------
    StandardScaler
        The loaded StandardScaler.
    """
    return joblib.load(path)

def prepare_input_data(raw_data, feature_columns, scaler_X, timesteps):
    """
    Apply the same feature engineering and scaling as during training.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw price/time-series data with columns like 'Close', 'Volume', etc.
    feature_columns : list
        Columns used as model inputs.
    scaler_X : fitted scaler
        Scaler used to normalize X during training.
    timesteps : int
        Number of timesteps the model expects per sample.
    """
    # Feature engineering (same as training)
    data = feature_engineering(raw_data)
    
    # Select same features and scale
    X = data[feature_columns].values
    X_scaled = scaler_X.transform(X)
    
    # Reshape to (1, timesteps, features) for prediction
    X_input = X_scaled[-timesteps:].reshape(1, timesteps, len(feature_columns))
    return X_input

def predict_next_close(model_path, scaler_y_path, scaler_X_path, raw_data, feature_columns, timesteps=10):
    """
    Predict the next-day close price using the trained model and latest data.
    """
    print("[INFO] Loading model and scalers...")
    model = load_model(model_path)
    y_scaler = load_scaler(scaler_y_path)
    X_scaler = load_scaler(scaler_X_path)

    # Prepare input
    X_input = prepare_input_data(raw_data, feature_columns, X_scaler, timesteps)

    # Predict next-day return
    y_pred_scaled = model.predict(X_input)
    y_pred_return = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]

    # Last known close
    last_close = raw_data["Close"].iloc[-1]

    # Reconstruct next-day predicted close
    pred_close_next = last_close * (1 + y_pred_return)

    print("=" * 60)
    print(f"Last Close: {last_close:.2f}")
    print(f"Predicted Return: {y_pred_return*100:.3f}%")
    print(f"Predicted Next Close: {pred_close_next:.2f}")
    print("=" * 60)

    return {
        "last_close": last_close,
        "predicted_return": y_pred_return,
        "predicted_close": pred_close_next
    }

if __name__ == "__main__":
    # Example usage
    model_path = MODEL_DIR / "best_model.keras"
    
    # Example: Load your latest stock data (replace with your own loader)
    data = pd.read_csv("data/latest_data.csv")

    feature_columns = [
        "Close", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
        "Daily_Return", "Volume"  # whatever you used during training
    ]
    
    results = predict_next_close(
        model_path=model_path,
        scaler_y_path=SCALER_Y_PATH,
        scaler_X_path=SCALER_X_PATH,
        raw_data=data,
        feature_columns=feature_columns,
        timesteps=10
    )

    print(results)
