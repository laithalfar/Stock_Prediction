import os
import sys

# add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50

# Model choice
MODEL_TYPE = "lstm"   # options: "lstm", "rnn"

# Paths
MODEL_DIR = "../models"

TRAINING_HISTORY_PATH = "../models/training_history.npy"

PLOT_PATH = "../models/training_plot.png"
