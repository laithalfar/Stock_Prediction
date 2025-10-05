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
MODEL_TYPE = "lstm"   # options: "lstm", "rnn", "cnn_gru"

TRAIN_PATH = project_root / "models/data/train.pkl"
TEST_PATH = project_root / "models/data/test.pkl"
SCALER_X_PATH = project_root / "models/Scalers/Standard_scaler_X.pkl"
SCALER_Y_PATH = project_root / "models/Scalers/Standard_scaler_y.pkl"

TRAINING_HISTORY_PATH = project_root / "models/history"

PLOT_PATH = project_root /"reports/plots"

