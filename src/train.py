"""
LSTM Stock Prediction Training Pipeline
=====================================
Main training script for LSTM-based stock price prediction models.
Handles data loading, preprocessing, model training, and evaluation.
"""

import numpy as np
import os
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import load_model
from data.processed import load_preprocessed_data
from config import TRAIN_PATH, TEST_PATH
from sklearn.preprocessing import StandardScaler
from src.model import create_lstm_model, create_rnn_model, create_cnn_gru_model
import kerastuner as kt

#Add project root to Python path
sys.path.append(os.path.abspath(".."))

# Import project modules
from config import LEARNING_RATE, BATCH_SIZE, EPOCHS, MODEL_TYPE, MODEL_DIR, TRAINING_HISTORY_PATH , PLOT_FOLD_PATH
from src.model import create_lstm_model
from data.processed import process_data
from src.data_preprocessing import split_train_val



# Ensure reproducibility
def setup_directories():
    """Create necessary directories for model saving."""

    (MODEL_DIR).mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / f"models_folds/{MODEL_TYPE}_folds").mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / f"history/{MODEL_TYPE}_history").mkdir(parents=True, exist_ok=True)
    (PLOT_FOLD_PATH).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created directories under: {MODEL_DIR}")


# Configure the callbacks
# Callbacks are functions that run automatically during training at specific points (like after each epoch or batch).
# They let you monitor, adjust, or stop training without manually intervening.
def setup_callbacks(fold):
    """Configure training callbacks."""
    # create models directory inside your project if it doesn't exist
    model_path = MODEL_DIR / f"models_folds/{MODEL_TYPE}_folds/model_fold_{fold+1}.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_loss", # monitor validation loss to stop training when it stops improving to avoid overfitting
            patience=7, # number of epochs with no improvement after which training will be stopped
            restore_best_weights=True, # restore model weights from the epoch with the best value of the monitored quantity
            verbose=1 # print messages when stopping
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_loss", # monitor validation loss to save the best model
            save_best_only=True, # only save the model if the monitored quantity has improved
            verbose=1,  # print messages when saving
        )
    ]

    return callbacks

# Load a saved model for a specific fold
def load_fold_model(fold: int):

    """
    Load a saved model for a specific fold.

    Parameters
    ----------
    fold : int
        The fold number for which to load the model.

    Returns
    -------
    keras.Model
        The loaded model.
    """

    filepath = MODEL_DIR / f"models_folds/{MODEL_TYPE}_folds/model_fold_{fold}.keras"

    if not filepath.exists():
        raise FileNotFoundError(f"Model file missing: {filepath}")
    return load_model(filepath)

# Load training history for a specific fold
def load_training_history(fold):
    """Load training history dict for a given fold."""
    history_path = TRAINING_HISTORY_PATH / f"history_fold_{fold+1}.npy"

    if not history_path.exists():
        raise FileNotFoundError(f"No history found for fold {fold+1}")
    return np.load(history_path, allow_pickle=True).item()


# Load the data
def load_and_preprocess_data(use_cached=True):
   
    """Load preprocessed data from cache if available, else run full pipeline."""
    if use_cached and os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        print("[INFO] Loading cached processed data...")
        return load_preprocessed_data()
    else:
        print("[INFO] No cached data found, running full preprocessing pipeline...")
        # Try fetching the data from the preprocessing module
        try:

            # Run the full preprocessing pipeline
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
                "feature_columns_train": preprocessed_data["feature_columns_train"],
                "Close_series": preprocessed_data["Close_series"]
            }
            
            return data
            
        except Exception as e:
            print(f"[ERROR] Data loading failed: {e}")
            raise



# Create the chosen model with hyperparameter tuning
def create_model(input_shape, model_type, X_tr, y_tr_scaled, validation_data, epochs, batch_size, callbacks):
    
    """
    Create a model of the chosen type with hyperparameter tuning.

    Parameters:
    - input_shape: tuple of shape of input data (timesteps, features)
    - model_type: string of the type of model to create (lstm, rnn, cnn_gru)
    - X_tr: numpy array of training data
    - y_tr_scaled: numpy array of scaled training data
    - validation_data: tuple of (X_val, y_val) for validation
    - epochs: int of number of epochs to train model
    - batch_size: int of batch size to use while training
    - callbacks: list of callbacks to use while training

    Returns:
    - model: the created model
    """
    print(f"[INFO] Creating {model_type.upper()} model...")

    # Define tuner for LSTM
    tuner_lstm = kt.RandomSearch(
        lambda hp: create_lstm_model(hp, input_shape),
        objective = "val_loss", # minimize validation loss
        max_trials = 10, # maximum number of different hyperparameter combinations to try
        executions_per_trial = 2, # number of times to train each model configuration to reduce variance
        directory = MODEL_DIR / "models_hyperparameters",
        project_name = "lstm_tuning"
    
    )

    # Example: run tuner for RNN
    tuner_rnn = kt.RandomSearch(
        lambda hp: create_rnn_model(hp, input_shape),
        objective = "val_loss", # minimize validation loss
        max_trials = 10, # maximum number of different hyperparameter combinations to try
        executions_per_trial = 2, # number of times to train each model configuration to reduce variance
        directory = MODEL_DIR / "models_hyperparameters",
        project_name = "rnn_tuning"
    )

    # Example: run tuner for CNN+GRU
    tuner_cnn_gru = kt.RandomSearch(
        lambda hp: create_cnn_gru_model(hp, input_shape),
        objective = "val_loss", # minimize validation loss
        max_trials = 10, # maximum number of different hyperparameter combinations to try
        executions_per_trial = 2, # number of times to train each model configuration to reduce variance
        directory = MODEL_DIR / "models_hyperparameters",
        project_name = "cnn_gru_tuning"
    )

    
    if model_type.lower() == "lstm":

        tuner = tuner_lstm
        # Fit hyperparameter tuning to data
        tuner.search(X_tr, y_tr_scaled, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks = callbacks)
        # Get the best hyperparameters
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best units:", best_hp.get("units_lstm1"))
        print("Best units:", best_hp.get("units_lstm2"))
        print("Best dropout:", best_hp.get("dropout_lstm1"))
        print("Best dropout:", best_hp.get("dropout_lstm2"))
        print("Best optimizer:", best_hp.get("optimizer_lstm"))

    elif model_type.lower() == "rnn":

        tuner = tuner_rnn
        # Fit hyperparameter tuning to data
        tuner.search(X_tr, y_tr_scaled, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks = callbacks)
        # Get the best hyperparameters
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best units_1:", best_hp.get("units_rnn1"))
        print("Best units_2:", best_hp.get("units_rnn2"))
        print("Best dropout:", best_hp.get("dropout_rnn1"))
        print("Best dropout:", best_hp.get("dropout_rnn2"))
        print("Best optimizer:", best_hp.get("optimizer_rnn"))

    elif model_type.lower() == "cnn_gru":

        tuner = tuner_cnn_gru
        # Fit hyperparameter tuning to data
        tuner.search(X_tr, y_tr_scaled, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks = callbacks)
        # Get the best hyperparameters
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best units:", best_hp.get("gru_units"))
        print("Best dropout:", best_hp.get("cnn_dropout"))
        print("Best optimizer:", best_hp.get("optimizer_cnn_gru"))

    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")
    
    print(f"[INFO] Model created with input shape: {input_shape}")
    model = tuner.get_best_models(num_models=1)[0]
    model.summary()
    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val, callbacks):
    
    
    """
    Train the model with the given hyperparameters.

    Parameters
    ----------
    model : keras.Model
        The model to be trained.
    X_train : numpy.ndarray
        The training data.
    y_train : numpy.ndarray
        The training labels.
    X_val : numpy.ndarray
        The validation data.
    y_val : numpy.ndarray
        The validation labels.
    callbacks : list
        List of callbacks for early stopping and model saving.

    Returns
    -------
    history : keras.callbacks.History
        The training history.
    """
    print("[INFO] Starting model training...")
    print(f"[INFO] Training parameters:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    try:
        history = model.fit(
            X_train, y_train, # Training data
            validation_data=(X_val, y_val), # Validation data for monitoring
            epochs=EPOCHS, # Total epochs to train
            batch_size=BATCH_SIZE, # Number of samples per gradient update
            callbacks=callbacks, # Callbacks for early stopping and model saving
            verbose=1, # Verbosity mode
            shuffle=False  # Important for time series data
        )
        
        print("[INFO] Training completed successfully!")
        return history
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise

# Walk forward validation function
def walk_forward_validation(X, y, train_window=350 , test_window=15):
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
            X[end_train:end_test], y[end_train:end_test],
            end_train, end_test
        )



# Save the training history
def save_training_history(history, fold):
    

    """
    Save the training history of a model to a file.

    Parameters
    ----------
    history : keras.callbacks.History
        The training history object.
    fold : int
        The fold number for which to save the training history.

    Returns
    -------
    str
        The path to the saved training history file.
    """
    training_history_path = TRAINING_HISTORY_PATH / f"history_fold_{fold+1}.npy"

    fold = fold + 1
    np.save(training_history_path, history.history)
    print(f"[INFO] Training history saved to: {training_history_path}")
    
    return training_history_path

def plot_training_history(history, fold, force_refresh=False):
    
    """
    Plot the training history of a model.

    Parameters
    ----------
    history : keras.callbacks.History or dict
        The training history object or a dict containing the history data.
    fold : int
        The fold number for which to plot the training history.
    force_refresh : bool, optional
        Whether to replot the training history even if a plot already exists.

    Returns
    -------
    str
        The path to the saved training history plot.
    """

    plot_path = PLOT_FOLD_PATH / f"plot_fold_{fold+1}.png"
    if plot_path.exists() and not force_refresh:
        print(f"[INFO] Plot already exists for fold {fold+1}: {plot_path}")
        return plot_path

    # Handle both cases
    if hasattr(history, "history"):  # Keras History object
        hist_data = history.history
    elif isinstance(history, dict):  # already a dict
        hist_data = history
    else:
        raise TypeError(f"Unsupported history type: {type(history)}")

    plt.figure(figsize=(12, 4))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(hist_data['loss'], label='Training Loss')
    plt.plot(hist_data['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot learning rate if available
    if 'lr' in hist_data:
        plt.subplot(1, 2, 2)
        plt.plot(hist_data['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.legend()

    plt.tight_layout()
    plot_path = PLOT_FOLD_PATH / f"plot_fold_{fold+1}.png"
    plt.savefig(PLOT_FOLD_PATH)
    plt.show()
    print(f"[INFO] Training plots saved to: {plot_path}")

# Main function
def train_pipeline():

    
    """
    Main training function for LSTM-based stock price prediction models.

    This function runs the walk-forward training pipeline for an LSTM model on a given dataset.
    It trains a new model for each fold of the walk-forward validation, saves the training history, and returns the trained models and their corresponding corresponding training histories.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Dictionary containing the trained models and their corresponding training histories.
    """

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
        X_te_list, y_te_scaled_list, y_tr_scaled_list, X_tr_list, close_te_list, y_scaler_list = [], [], [], [], [], []

        # Setup output variables
        model = []
        history = []

        # Get input shape
        input_shape = (X.shape[1], X.shape[2])

        # Fold is a counter for each walk-forward iteration
        # Each iteration trains on a rolling window and tests on the subsequent window
        # This simulates real-world sequential prediction and the iteration is done using the enumerate function
        for fold, (X_tr, y_tr, X_te, y_te, end_train, end_test) in enumerate(
            walk_forward_validation(X, y, train_window=252, test_window=21)
        ):
            # Define model path for this fold
            model_path = MODEL_DIR / f"models_folds/{MODEL_TYPE}_folds/model_fold_{fold+1}.keras"

            # Scale y values (important for regression tasks)
            # We fit the scaler only on the training data to avoid data leakage
            '''When you train a neural network (especially with regression targets like stock prices), you want both your inputs (X) and outputs (y) to 
            live in numerically stable ranges — typically somewhere around 0 to 1, or -1 to 1.
            
            Neural networks, particularly those with activations like ReLU or tanh, train more smoothly when their inputs and outputs aren’t vastly different in scale.

            So StandardScaler rescales your targets y to have:

            mean = 0

            standard deviation = 1

            StandardScaler expects a 2D array of shape (n_samples, n_features) — even if you only have one feature (y).
            reshape(-1, 1) converts a 1D array like [y₁, y₂, …] into column form.

            After scaling, .ravel() flattens it back to 1D for convenience.
            '''
            y_scaler = StandardScaler()
            y_tr_scaled = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
            y_te_scaled = y_scaler.transform(y_te.reshape(-1, 1)).ravel()

            # Check if model already exists for this fold
            if model_path.exists():
                print(f"[INFO] Found existing model for fold {fold+1}, loading...")
                
                # Load model and history if they exist
                loaded_model = load_model(model_path)
                model.append(loaded_model)
                history.append(load_training_history(fold))

            else:
                print(f"[INFO] Training new model for fold {fold+1}...")
               
            
                print(f"\n[INFO] Fold {fold+1}: Train={len(X_tr)}, Test={len(X_te)}")

                # Test training window into train+val (time-ordered)
                X_tr, y_tr_scaled, X_val, y_val = split_train_val(X_tr, y_tr_scaled, val_frac=0.2)                

                # Create fresh model for each fold
                callbacks = setup_callbacks(fold)

                # Train new model each time to avoid data leakage
                model.append(create_model(input_shape, MODEL_TYPE, X_tr, y_tr_scaled, validation_data=(X_val, y_val), epochs = EPOCHS, batch_size = BATCH_SIZE, callbacks = callbacks))
                  

                # Train with different X_train, y_train, X_val, y_val each time
                history.append(train_model(model[fold], X_tr, y_tr_scaled, X_val, y_val, callbacks))

                # Save training history
                save_training_history(history[fold], fold)

                # Save model path instead of model itself to reduce memory usage
                model_path = MODEL_DIR / f"models_folds/{MODEL_TYPE}_folds/model_fold_{fold+1}.keras"
                model[fold].save(model_path)
                print(f"[INFO] Saved model for fold {fold+1} to {model_path}")


            # Save x_te and y_te for evaluation
            X_te_list.append(X_te)
            y_te_scaled_list.append(y_te_scaled) 
            y_tr_scaled_list.append(y_tr_scaled)
            X_tr_list.append(X_tr)
            close_te_list.append(preprocessed_data["Close_series"][end_train : end_test + 1])
            y_scaler_list.append(y_scaler)


        # Return traininig results 
        train_results = {
            "model_list": model,
            "history_list": history,
            "X_te_list": X_te_list,
            "y_te_scaled_list": y_te_scaled_list,
            "y_tr_scaled_list": y_tr_scaled_list,
            "X_tr_list": X_tr_list,
            "feature_columns": feature_columns,
            "close_te_list": close_te_list,
            "y_scaler_list": y_scaler_list
        }


        return train_results

    except Exception as e:
        print(f"[CRITICAL ERROR] Walk-forward training failed: {e}")
        raise

