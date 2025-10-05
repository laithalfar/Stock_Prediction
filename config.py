import os
import sys
from pathlib import Path

# add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50

project_root = Path(__file__).resolve().parent.parent / "stock-forecast"

# Paths
MODEL_DIR = project_root / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)   # ensure directory exists

# Model choice
MODEL_TYPE = "cnn_gru"   # options: "lstm", "rnn", "cnn_gru"

TRAIN_PATH = project_root / "models/data/train.pkl"
TEST_PATH = project_root / "models/data/test.pkl"
SCALER_X_PATH = project_root / f"models/Scalers/{MODEL_TYPE}_scalers/Standard_scaler_X.pkl"
SCALER_Y_PATH = project_root / f"models/Scalers/{MODEL_TYPE}_scalers/Standard_scaler_y.pkl"

TRAINING_HISTORY_PATH = project_root / f"models/history/{MODEL_TYPE}_history"

PLOT_PATH = project_root /f"reports/plots/{MODEL_TYPE}_plots"

