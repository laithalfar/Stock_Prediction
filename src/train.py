"""
LSTM Stock Prediction Training Pipeline
=====================================
Main training script for LSTM-based stock price prediction models.
Handles data loading, preprocessing, model training, and evaluation.
"""

import numpy as np
import os
import sys
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt



#Add project root to Python path
sys.path.append(os.path.abspath(".."))

# Import project modules
from config import LEARNING_RATE, BATCH_SIZE, EPOCHS, MODEL_TYPE, MODEL_DIR, TRAINING_HISTORY_PATH , PLOT_PATH
from src.model import create_lstm_model, create_recurrent_neural_network, create_gru_model
from data.processed import process_data
from src.data_preprocessing import split_train_val
import joblib



# Ensure reproducibility
def setup_directories():
    """Create necessary directories for model saving."""

    (MODEL_DIR).mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "models_folds").mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "history").mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "plots").mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created directories under: {MODEL_DIR}")


# Configure the callbacks
# Callbacks are functions that run automatically during training at specific points (like after each epoch or batch).
# They let you monitor, adjust, or stop training without manually intervening.
def setup_callbacks(fold):
    """Configure training callbacks."""
    # create models directory inside your project if it doesn't exist
    model_path = MODEL_DIR / f"models_folds/model_fold_{fold}.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
    ]

    return callbacks

#load the data
def load_and_preprocess_data():
    """Load and preprocess data for training."""
    print("[INFO] Loading and preprocessing data...")
    
    # Try fetching the data from the preprocessing module
    try:
        preprocessed_data = process_data()
        
        # Validate data shapes
        if len(preprocessed_data["X_train"].shape) != 3:
            raise ValueError(f"Expected 3D input for LSTM, got {preprocessed_data["X_train"].shape}")
        
        # Get information on each dataset
        print(f"[INFO] Data shapes:")
        print(f"  Training: X{preprocessed_data["X_train"].shape}, y{preprocessed_data["y_train"].shape}")
        print(f"  Test: X{preprocessed_data["X_test"].shape}, y{preprocessed_data["y_test"].shape}")

        # Return all datasets in a dictionary
        data={
            "X_train": preprocessed_data["X_train"],
            "y_train": preprocessed_data["y_train"],
            "X_test": preprocessed_data["X_test"],
            "y_test": preprocessed_data["y_test"],
            "feature_columns_train": preprocessed_data["feature_columns_train"]
        }
        
        return data
        
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        raise

# Create the model
def create_model(input_shape, model_type):
    """Create and configure the specified model."""
    print(f"[INFO] Creating {model_type.upper()} model...")
    
    if model_type.lower() == "lstm":
        model = create_lstm_model(input_shape)
    elif model_type.lower() == "rnn":
        model = create_recurrent_neural_network(input_shape)
    elif model_type.lower() == "gru":
        model = create_gru_model(input_shape)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")
    
    print(f"[INFO] Model created with input shape: {input_shape}")
    model.summary()
    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val, callbacks):
    """Train the model with given data and callbacks."""
    print("[INFO] Starting model training...")
    print(f"[INFO] Training parameters:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series data
        )
        
        print("[INFO] Training completed successfully!")
        return history
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise

# Walk forward validation function
def walk_forward_validation(X, y, train_window=252, test_window=21):
    """
    Perform walk-forward validation for time series models.
    
    Args:
        X: Feature array
        y: Target array
        train_window: Number of samples for training (default: ~1 year of trading days)
        test_window: Number of samples for testing (default: ~1 month of trading days)
    
    Yields:
        Tuple of (X_train, y_train, X_test, y_test) for each test
    """
    n_samples = len(X)
    
    for start in range(0, n_samples - train_window - test_window + 1, test_window):
        end_train = start + train_window
        end_test = end_train + test_window
        
        if end_test > n_samples:
            break

        # The yield keyword in Python is used to create generator functions. Unlike regular functions that use return to send a value and terminate,
        # generator functions use yield to produce a sequence of values one at a time,
        # pausing and resuming their execution.   
        yield (
            X[start:end_train], y[start:end_train],
            X[end_train:end_test], y[end_train:end_test]
        )

# Save the training history
def save_training_history(history, fold):
    """Save training history for analysis."""

    training_history_path = TRAINING_HISTORY_PATH / f"history_fold_{fold+1}.npy"
    fold = fold + 1
    np.save(training_history_path, history.history)
    print(f"[INFO] Training history saved to: {training_history_path}")
    
    return training_history_path


# Plot training history
def plot_training_history(history, fold):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate if available
    if 'lr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.legend()
    
    plt.tight_layout()
    plot_path = PLOT_PATH / f"plot_fold_{fold+1}.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"[INFO] Training plots saved to: {plot_path}")


# Main function
def train_pipeline():

    """Main training pipeline using walk-forward validation only."""
    print("="*60)
    print("LSTM STOCK PREDICTION TRAINING PIPELINE (Walk-Forward)")
    print("="*60)

    try:

        # Create necessary directories
        setup_directories()

        # Load full dataset
        preprocessed_data = load_and_preprocess_data()
        X_train = preprocessed_data["X_train"]
        X_test  = preprocessed_data["X_test"]

        y_train = preprocessed_data["y_train"]
        y_test  = preprocessed_data["y_test"]

        feature_columns = preprocessed_data["feature_columns_train"]
      
        # Merge everything into one sequence (important for walk-forward)
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        # Set input variables
        X_te_list, y_te_list, X_tr_list = [], [], []

        # Setup output variables
        model = []
        history = []

        # Get input shape
        input_shape = (X.shape[1], X.shape[2])

        # Fold is a counter for each walk-forward iteration
        # Each iteration trains on a rolling window and tests on the subsequent window
        # This simulates real-world sequential prediction and the iteration is done using the enumerate function
        for fold, (X_tr, y_tr, X_te, y_te) in enumerate(
            walk_forward_validation(X, y, train_window=252, test_window=21)
        ):
            
            print(f"\n[INFO] Fold {fold+1}: Train={len(X_tr)}, Test={len(X_te)}")

            # test training window into train+val (time-ordered)
            X_tr, y_tr, X_val, y_val = split_train_val(X_tr, y_tr, val_frac=0.2)

            # Create fresh model for each fold
            model.append(create_model(input_shape, MODEL_TYPE))
            callbacks = setup_callbacks(fold)  

            # Train with different X_train, y_train, X_val, y_val each time
            history.append(train_model(model[fold], X_tr, y_tr, X_val, y_val, callbacks))

            # Save training history
            save_training_history(history[fold], fold)

            # Save x_te and y_te for evaluation
            X_te_list.append(X_te)
            y_te_list.append(y_te) 
            X_tr_list.append(X_tr)

            # Save model path instead of model itself to reduce memory usage
            model_path = MODEL_DIR / f"models_folds/model_fold_{fold+1}.keras"
            model[fold].save(model_path)
            print(f"[INFO] Saved model for fold {fold+1} to {model_path}")
            

        # Return traininig results 
        train_results = {
            "model_list": model,
            "history_list": history,
            "X_te_list": X_te_list,
            "y_te_list": y_te_list,
            "X_tr_list": X_tr_list,
            "feature_columns": feature_columns
        }


        return train_results

    except Exception as e:
        print(f"[CRITICAL ERROR] Walk-forward training failed: {e}")
        raise

