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
MODEL_TYPE = "lstm"   # options: "lstm", "rnn"

TRAIN_PATH = project_root / "models/data/train.pkl"
VAL_PATH = project_root / "models/data/val.pkl"
TEST_PATH = project_root / "models/data/test.pkl"

TRAINING_HISTORY_PATH = project_root / "models/training_history"

PLOT_PATH = project_root /"notebooks/plots"

