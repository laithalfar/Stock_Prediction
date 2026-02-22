import sys
import os

# Add project root to sys.path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import
from src.data_preprocessing import load_data, feature_engineering, clean_data, preprocess
import joblib
from config import DATA_PATH


# Save processed data 
def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, folds , y_scaler_list, X_scaler_list, feature_columns_X, close):
    
    
    """
    Save processed data to disk using joblib.

    Parameters
    ----------
    X_train : array-like
        Training input data
    y_train : array-like
        Training output data
    X_val : array-like
        Validation input data
    y_val : array-like
        Validation output data
    X_test : array-like
        Testing input data
    y_test : array-like
        Testing output data
    close_list : list
        List of Close prices
    fold : int
        Fold number for walk-forward validation
    y_scaler_list : list
        List of StandardScaler objects for y
    X_scaler_list : list
        List of StandardScaler objects for X
    feature_columns_X : list
        List of feature columns for X

    Returns
    -------
    None
    """
    joblib.dump((X_train, y_train, X_val, y_val, X_test, y_test, folds , y_scaler_list, X_scaler_list, feature_columns_X, close), DATA_PATH)
    print(f"[INFO] Processed data saved to {DATA_PATH} ")

# Load splits
def load_preprocessed_data():
    
    """
    Load processed data from disk using joblib.

    Returns
    -------
    dict
        Dictionary containing the X and y data, as well as the feature columns and Close series.
    """
    
    X_train, y_train, X_val, y_val, X_test, y_test, folds , y_scaler_list, X_scaler_list, feature_columns_X, close = joblib.load(DATA_PATH)
    print(f"[INFO] Processed data loaded from {DATA_PATH}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "close_te_list": close,
        "folds": folds,
        "y_scaler_list": y_scaler_list,
        "X_scaler_list":  X_scaler_list,
        "feature_columns_X": feature_columns_X
    }


# Process the data
def process_data():
    
    """
    Process the data by loading, cleaning, feature engineering, and splitting it into training, testing, and validation sets.

    Returns
    -------
    dict
        Dictionary containing the training, testing, and validation data, as well as the feature columns and Close series.
    """
    # Load the data
    df = load_data()

    # Clean the data
    df = clean_data(df)

    # Feature engineering
    df, close  = feature_engineering(df)

    # Drop empty data
    df = df.dropna()
    
    # Split data into training, testing, and validation sets
    data = preprocess(df, 'Next_Day_Return', close)

    # 🔽 Save processed splits here
    save_processed_data(data["X_train"], data["y_train"], data["X_val"], data["y_val"], data["X_test"], data["y_test"], data["folds"] ,data["y_scaler_list"], data["X_scaler_list"], data["feature_columns_X"], close) 
                        
    return data

