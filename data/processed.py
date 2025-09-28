import pandas as pd
import sys
import os

# Add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))

# Import
from src.data_preprocessing import load_data, feature_engineering, clean_data, splitting_data
import joblib
from config import TRAIN_PATH, TEST_PATH

# Save splits

def save_processed_data(X_train, y_train, X_test, y_test, feature_columns):
    """Save processed datasets and feature columns."""
    joblib.dump((X_train, y_train, feature_columns), TRAIN_PATH)
    joblib.dump((X_test, y_test), TEST_PATH)
    print(f"[INFO] Processed data saved to {TRAIN_PATH} and {TEST_PATH}")

# Load splits
def load_processed_data():
    """Load processed datasets and feature columns."""
    X_train, y_train, feature_columns = joblib.load(TRAIN_PATH)
    X_test, y_test = joblib.load(TEST_PATH)
    print(f"[INFO] Processed data loaded from {TRAIN_PATH} and {TEST_PATH}")
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_columns_train": feature_columns,
    }

# Process the data
def process_data():

    # Load the data
    df = load_data()

    # Clean the data
    df = clean_data(df)

    # Feature engineering
    df  = feature_engineering(df)

    # Drop empty data
    df = df.dropna()
    
    # Split data into training, testing, and validation sets
    data = splitting_data(df, 'Next_Day_Return')

    # 🔽 Save processed splits here
    save_processed_data(data["X_train"], data["y_train"], data["X_test"], data["y_test"], data["feature_columns_train"]) 

    return data

