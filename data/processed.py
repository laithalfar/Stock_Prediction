import pandas as pd
import sys
import os

# add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))

#import
from src.data_preprocessing import load_data, feature_engineering, clean_data, splitting_data
import joblib
from config import TRAIN_PATH, VAL_PATH, TEST_PATH

#save splits
def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """Save processed datasets into /data/processed directory."""
    joblib.dump((X_train, y_train), TRAIN_PATH)
    joblib.dump((X_val, y_val), VAL_PATH)
    joblib.dump((X_test, y_test), TEST_PATH)


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
    data = splitting_data(df, 'Next_Day_Return')

    # 🔽 Save processed splits here
    save_processed_data(data["X_train"], data["y_train"], data["X_val"], data["y_val"], data["X_test"], data["y_test"]) 

    return data

