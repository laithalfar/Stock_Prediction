import pandas as pd
import sys
import os

# Add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))

# Import
from src.data_preprocessing import load_data, feature_engineering, clean_data, splitting_data
import joblib
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

# Save splits
def save_processed_data(X_train, y_train, X_test, y_test):
    """Save processed datasets into /data/processed directory."""
    joblib.dump((X_train, y_train), TRAIN_PATH)
    joblib.dump((X_test, y_test), TEST_PATH)


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
    save_processed_data(data["X_train"], data["y_train"], data["X_split"], data["y_split"]) 

    return data

