import os
import sys
from pathlib import Path

# Project root = the directory this file lives in
project_root = Path(__file__).resolve().parent

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20   # capped — EarlyStopping handles early exit; 20 is enough for small folds

# Model choice
MODEL_TYPE = "lstm"   # options: "lstm", "rnn", "cnn_gru"

# Paths
MODEL_DIR = project_root / "models"
DATA_PATH = MODEL_DIR / "data" / "preprocessed_data.pkl"
SCALER_X_PATH = MODEL_DIR / f"Scalers/{MODEL_TYPE}_scalers/Standard_scaler_X.pkl"
SCALER_Y_PATH = MODEL_DIR / f"Scalers/{MODEL_TYPE}_scalers/Standard_scaler_y.pkl"
TRAINING_HISTORY_PATH = MODEL_DIR / f"history/{MODEL_TYPE}_history"
PLOT_FOLD_PATH = project_root / f"reports/plots/{MODEL_TYPE}_plots"
PLOT_ACTUAL_PREDICTED_PATH = project_root / f"reports/{MODEL_TYPE}_Actual_predicted_plots.png"

# Ensure all directories exist
for _dir in [
    MODEL_DIR / "data",
    MODEL_DIR / f"models_folds/{MODEL_TYPE}_folds",
    MODEL_DIR / f"history/{MODEL_TYPE}_history",
    MODEL_DIR / f"results/{MODEL_TYPE}_results",
    MODEL_DIR / "results",
    MODEL_DIR / f"Scalers/{MODEL_TYPE}_scalers",
    PLOT_FOLD_PATH,
    project_root / "reports",
]:
    _dir.mkdir(parents=True, exist_ok=True)
