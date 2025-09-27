import pandas as pd
import sys
import os

# add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))

#import
from src.data_preprocessing import load_data, feature_engineering, clean_data, splitting_data
import joblib


#save splits
def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """Save processed datasets into /data/processed directory."""
    joblib.dump((X_train, y_train), "../models/train.pkl")
    joblib.dump((X_val, y_val), "../models/val.pkl")
    joblib.dump((X_test, y_test), "../models/data/test.pkl")


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
    X_train, X_test, y_train, y_test, X_val, y_val, feature_columns_train = splitting_data(df, 'Next_Day_Return')

    # 🔽 Save processed splits here
    save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test)

    return X_train, X_test, y_train, y_test, X_val, y_val, feature_columns_train

