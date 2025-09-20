import pandas as pd
import sys
import os

# add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))

#import
from src.data_preprocessing import load_data, feature_engineering, clean_data, data_transformation, splitting_data, min_max_scale
import joblib


#save splits
def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """Save processed datasets into /data/processed directory."""
    joblib.dump((X_train, y_train), "data/processed/train.pkl")
    joblib.dump((X_val, y_val), "data/processed/val.pkl")
    joblib.dump((X_test, y_test), "data/processed/test.pkl")

#save scalers
def save_scaler_data(min_max_scaler):
    """Save processed datasets into /data/processed directory."""
    # 🔽 Save scalers here
    joblib.dump(min_max_scaler, "data/processed/min_max_scaler.pkl")

#process the data
def process_data():

    #load the data
    df = load_data()

    #clean the data
    df = clean_data(df)

    #feature engineering
    df  = feature_engineering(df)

    #drop empty data
    df = df.dropna()
    
    #split data into training, testing, and validation sets
    X_train, X_test, y_train, y_test, X_val, y_val = splitting_data(df, 'Close')

    # Fit and transform train
    X_train, min_max_scaler = min_max_scale(X_train)

    # Apply the same scaler to val and test
    X_val = min_max_scaler.transform(X_val)
    X_test = min_max_scaler.transform(X_test)

    #save scalers
    save_scaler_data(min_max_scaler)

    # 🔽 Save processed splits here
    save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test)

    return X_train, X_test, y_train, y_test, X_val, y_val

