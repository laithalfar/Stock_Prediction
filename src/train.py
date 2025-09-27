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
import pandas as pd


#Add project root to Python path
sys.path.append(os.path.abspath(".."))

# Import project modules
from config import LEARNING_RATE, BATCH_SIZE, EPOCHS, MODEL_TYPE, MODEL_DIR, TRAINING_HISTORY_PATH , PLOT_PATH
from src.data_preprocessing import align_features
from src.model import create_lstm_model, create_recurrent_neural_network, create_gru_model
from data.processed import process_data
import joblib



# Ensure reproducibility
def setup_directories():
    """Create necessary directories for model saving."""
    model_dir = os.path.dirname(MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    print(f"[INFO] Created directories: {model_dir}")

#load the data
def load_and_preprocess_data():
    """Load and preprocess data for training."""
    print("[INFO] Loading and preprocessing data...")
    
    #try fetching the data from the preprocessing module
    try:
        X_train, X_test, y_train, y_test, X_val, y_val, feature_columns_train= process_data()
        
        # Validate data shapes
        if len(X_train.shape) != 3:
            raise ValueError(f"Expected 3D input for LSTM, got {X_train.shape}")
        
        #get information on each dataset
        print(f"[INFO] Data shapes:")
        print(f"  Training: X{X_train.shape}, y{y_train.shape}")
        print(f"  Validation: X{X_val.shape}, y{y_val.shape}")
        print(f"  Test: X{X_test.shape}, y{y_test.shape}")

        # Return all datasets in a dictionary
        preprocessed_data={
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns_train": feature_columns_train
        }
        
        return preprocessed_data
        
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


# Configure the callbacks
# Callbacks are functions that run automatically during training at specific points (like after each epoch or batch).
# They let you monitor, adjust, or stop training without manually intervening.
def setup_callbacks(fold):
    """Configure training callbacks."""
    model_path = os.path.join(MODEL_DIR, f"model_fold_{fold+1}.h5")
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
        Tuple of (X_train, y_train, X_test, y_test) for each split
    """
    n_samples = len(X)
    
    for start in range(0, n_samples - train_window - test_window + 1, test_window):
        end_train = start + train_window
        end_test = end_train + test_window
        
        if end_test > n_samples:
            break
            
        yield (
            X[start:end_train], y[start:end_train],
            X[end_train:end_test], y[end_train:end_test]
        )

# Save the training history
def save_training_history(history):
    """Save training history for analysis."""
    training_history_path = TRAINING_HISTORY_PATH
    np.save(training_history_path, history.history)
    print(f"[INFO] Training history saved to: {training_history_path}")
    
    return training_history_path


# Plot training history
def plot_training_history(history):
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
    plot_path = PLOT_PATH
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
        preprocessed_data= load_and_preprocess_data()
        X_train = preprocessed_data["X_train"]
        X_val   = preprocessed_data["X_val"]
        X_test  = preprocessed_data["X_test"]

        y_train = preprocessed_data["y_train"]
        y_val   = preprocessed_data["y_val"]
        y_test  = preprocessed_data["y_test"]

        feature_columns = preprocessed_data["feature_columns_train"]
      
        # Merge everything into one sequence (important for walk-forward)
        X = np.concatenate([X_train, X_val, X_test])
        y = np.concatenate([y_train, y_val, y_test])

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

            # Create fresh model for each fold
            model.append(create_model(input_shape, MODEL_TYPE))
            callbacks = setup_callbacks(fold)  

            # Train with different x_train, y_train, x_test, y_test each time
            history.append(train_model(model[fold], X_tr, y_tr, X_te, y_te, callbacks))

            # Save x_te and y_te for evaluation
            X_te_list.append(X_te)
            y_te_list.append(y_te) 
            X_tr_list.append(X_tr)

            # Save model path instead of model itself to reduce memory usage
            model_path = f"../models/model_fold_{fold+1}.h5"
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

