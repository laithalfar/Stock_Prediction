import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib
import os
import sys


# add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))


#extract stock data from yfinance
def yfinance_data_to_excel(ticker, period, interval):
        
    #get apple data
    dat = yf.Ticker(ticker)

    #place apple data in a dataframe variables
    OHCL = dat.history(period= period, interval = interval) # get OHCL data
    General_info = pd.DataFrame([dat.info]) # get general info data
    analyst_price_targets = pd.DataFrame([dat.analyst_price_targets]) # get the predictions of analysts for OHCL in 12-18months
    quarterly_income_stmt =  dat.quarterly_income_stmt # get the quarterly income statement
    quarterly_balance_sheet = dat.quarterly_balance_sheet # get the quarterly balance sheet
    quarterly_cashflow = dat.quarterly_cashflow # get the quarterly cashflow


    # excel does not support timezones so timezobes are removed prior
    OHCL.index = OHCL.index.tz_localize(None)


    #save the data in a excel file in different sheets for better viewing and analyses
    with pd.ExcelWriter("data/raw.xlsx") as writer:
        OHCL.to_excel(writer, sheet_name=f"{ticker}_OHCL")
        General_info.to_excel(writer, sheet_name=f"{ticker}_General_info")
        analyst_price_targets.to_excel(writer, sheet_name=f"{ticker}_analyst_price_targets")
        quarterly_income_stmt.to_excel(writer, sheet_name=f"{ticker}_quarterly_income_stmt")
        quarterly_balance_sheet.to_excel(writer, sheet_name=f"{ticker}_quarterly_balance_sheet")
        quarterly_cashflow.to_excel(writer, sheet_name=f"{ticker}_quarterly_cashflow")

        return 0

#safely load raw data
def load_data():

    #stock parameters
    ticker = "AAPL"
    period = "3y"
    interval = "1d"

    #get stock data
    yfinance_data_to_excel(ticker, period, interval)

    #file paramaters
    file = "data/raw.xlsx"
    sheet_name = f"{ticker}_OHCL"


    # Load data from excel file
    if file.endswith(".xlsx"):
        load_data = pd.read_excel(file, sheet_name= sheet_name)
    elif file.endswith(".csv"):
        load_data = pd.read_csv(file)
    else:
        raise ValueError("File format not supported. Please use .xlsx or .csv files.")

    return load_data

#  Clean data from duplicates and NaNs
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Fix missing values, duplicates, datatypes."""
    data = data.drop_duplicates()
    data = data.ffill()  # New syntax
    return data


# Process the data
def feature_engineering(data):

    # Calculate moving averages, and cumulative returns
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    

    # Calculate daily returns 
    data['Daily_Return'] = data['Close'].pct_change()
    
    
    # --- EMA (Exponential Moving Average) ---
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    # --- MACD (Moving Average Convergence Divergence) Histogram---
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']


    # --- RSI (Relative Strength Index) ---
    delta = data["Close"].diff() #difference between between this close and the one prior
    gain = delta.where(delta > 0, 0.0) #determine whether differences fall into gain or loss category
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean() #get moving average
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    avg_gain = avg_gain.ewm(com=14-1, adjust=False).mean()   # Use exponential moving average for smoother RSI
    avg_loss = avg_loss.ewm(com=14-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # 30-day rolling standard deviation of 'Close' aka volatility 
    data['STD_30'] = data['Close'].rolling(window=30).std()

    # Bollinger Bands parameters
    window = 20
    k = 2

    # --- Bollinger Bands ---
    data['BB_Middle'] = data['Close'].rolling(window=window).mean()
    data['BB_STD'] = data['Close'].rolling(window=window).std()
    data['BB_Upper'] = data['BB_Middle'] + k * data['BB_STD']
    data['BB_Lower'] = data['BB_Middle'] - k * data['BB_STD']
    data['BB_pct'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    #Support/Resistance
    data["Rolling_Max_20"] = data["Close"].rolling(20).max()
    data["Rolling_Min_20"] = data["Close"].rolling(20).min()
    data["Dist_To_Support"] = data["Close"] - data["Rolling_Min_20"]
    data["Dist_To_Resistance"] = data["Rolling_Max_20"] - data["Close"]

    #Daily Volatility
    data["Daily_Range"] = data["High"] - data["Low"]
    data["Range_Pct"] = (data["High"] - data["Low"]) / data["Close"]

    #candlestick body (bearish vs bullish)
    data["Candlestick_Body"] = data["Close"] - data["Open"]

    #GoldenCross and DeathCross
    data["GoldenCross"] = ((data["SMA_50"] > data["SMA_200"]) & (data["SMA_50"].shift(1) <= data["SMA_200"].shift(1))).astype(int)
    data["DeathCross"] = ((data["SMA_50"] < data["SMA_200"]) & (data["SMA_50"].shift(1) >= data["SMA_200"].shift(1))).astype(int)

    #MA spread and regime
    data["MA_Spread"] = data["SMA_50"] - data["SMA_200"]


    data['Next_Day_Return'] = data['Close'].pct_change().shift(-1)  # Tomorrow's return

    # --- Select Features ---
    features = ['SMA_50', 'SMA_200', 'Daily_Return','MACD_Histogram', 'RSI', 'BB_pct', 'STD_30', 'Dist_To_Support', 'Dist_To_Resistance', 'Candlestick_Body', 'MA_Spread', 'GoldenCross', 'DeathCross', 'Close', 'Volume', 'Range_Pct', 'Next_Day_Return']

    # Only include features that exist in the data
    available_features = [f for f in features if f in data.columns]
    X = data[available_features]  # Drop rows with NaN from rolling windows
    
    return X

# Min-Max scaling
def min_max_scale(data, columns=None):
    """Scale selected columns to [0,1]."""
    
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns.tolist()
    
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data, scaler  # return scaler to apply to test set

def one_hot_encode(data, columns=None):
    """
    One-hot encode categorical columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataframe.
    columns : list or None
        List of columns to encode. If None, automatically detect object or categorical columns.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one-hot encoded columns.
    """
    if columns is None:
        # Automatically detect categorical/object columns
        columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    data = pd.get_dummies(data, columns=columns, drop_first=True)
    return data


# Transform data
def data_transformation(data):
    """Transform data for model training."""

    #one hot encode categorical variables
    data = one_hot_encode(data)

     #scale data
    data, min_max_scaler = min_max_scale(data)

    return data, min_max_scaler
    
#change to 3D for lstm input
def create_lstm_input(data, feature_columns, timesteps=10):
    """
    Convert dataframe to 3D array for LSTM:
    [samples, timesteps, features].
    
    Parameters:
    - data: pandas DataFrame, already scaled/cleaned
    - feature_columns: list of column names to use as features
    - timesteps: number of previous steps to include
    
    Returns:
    - X: np.array of shape [samples, timesteps, features]
    """
    data = data[feature_columns].values
    X = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
    X = np.array(X)
    return X

#split features and targets into x and y respectively
def split_features_target(data, target_col):

    """Split data into features (X) and target (y)."""
    X = data.drop(columns=[target_col])
    y = data[target_col]

    return X, y

#make sure test and train data have same features
def align_features(X_df, train_columns):
    """Ensure DataFrame has the same features as training data."""
    for col in train_columns:
        if col not in X_df.columns:
            X_df[col] = 0
    for col in list(X_df.columns):
        if col not in train_columns:
            X_df = X_df.drop(col, axis=1)
    return X_df[train_columns]

#check feature alignment worked well
def check_feature_alignment(X_test, X_train):
    if X_test.shape[-1] != X_train.shape[-1]:
        raise ValueError(
            f"Mismatch in feature count: test={X_test.shape[-1]}, train={X_train.shape[-1]}"
        )
    return X_test


#save scalers
def save_scaler_data(min_max_scaler):
    """Save processed datasets into /data/processed directory."""
    # 🔽 Save scalers here
    joblib.dump(min_max_scaler, "../models/min_max_scaler.pkl")


#splitting data into training, val and testing sets
def splitting_data(data, target_col, timesteps=10):

    """Split data into training and testing sets."""
    X, y = split_features_target(data, target_col)

    # Time-based split (no shuffling!)
    train_size = int(len(X) * 0.7)  # 70% for training
    val_size = int(len(X) * 0.15)   # 15% for validation
    # Remaining 15% for testing
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]

    #Store column names BEFORE transformation
    feature_columns_train = X_train.columns.tolist()

    # Transform data
    X_train, min_max_scaler = data_transformation(X_train)

    #save scalers
    save_scaler_data(min_max_scaler)

    X_train = pd.DataFrame(X_train, columns=feature_columns_train)
    X_test = pd.DataFrame(min_max_scaler.transform(X_test), columns=feature_columns_train)
    X_val = pd.DataFrame(min_max_scaler.transform(X_val), columns=feature_columns_train)

    # Align features
    X_train = align_features(X_train, feature_columns_train)
    X_test = align_features(X_test, feature_columns_train)
    X_val = align_features(X_val, feature_columns_train)

    # Create LSTM input
    X_train = create_lstm_input(X_train, feature_columns_train, timesteps)
    X_test = create_lstm_input(X_test, feature_columns_train, timesteps)
    X_val = create_lstm_input(X_val, feature_columns_train, timesteps)

    check_feature_alignment(X_test, X_train)
    check_feature_alignment(X_val, X_train)
    
    # Adjust y arrays to match LSTM sequence length
    y_train = y_train[timesteps:].values
    y_val = y_val[timesteps:].values
    y_test = y_test[timesteps:].values
    
    
    return X_train, X_test, y_train, y_test, X_val, y_val, feature_columns_train
