import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib
import os
import sys


# Add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))


# Extract stock data from yfinance
def yfinance_data_to_excel(ticker, period, interval):
        
    # Get apple data
    dat = yf.Ticker(ticker)

    # Place apple data in a dataframe variables
    OHCL = dat.history(period= period, interval = interval) # get OHCL data
    General_info = pd.DataFrame([dat.get_info()]) # get general info data
    analyst_price_targets = pd.DataFrame([dat.analyst_price_targets]) # get the predictions of analysts for OHCL in 12-18months
    quarterly_income_stmt =  dat.quarterly_income_stmt.T # get the quarterly income statement
    quarterly_balance_sheet = dat.quarterly_balance_sheet.T # get the quarterly balance sheet
    quarterly_cashflow = dat.quarterly_cashflow.T # get the quarterly cashflow


    # Excel does not support timezones so timezobes are removed prior
    OHCL.index = OHCL.index.tz_localize(None)


    # Save the data in a excel file in different sheets for better viewing and analyses
    with pd.ExcelWriter("data/raw.xlsx") as writer:
        OHCL.to_excel(writer, sheet_name=f"{ticker}_OHCL")
        General_info.to_excel(writer, sheet_name=f"{ticker}_General_info")
        analyst_price_targets.to_excel(writer, sheet_name=f"{ticker}_analyst_price_targets")
        quarterly_income_stmt.to_excel(writer, sheet_name=f"{ticker}_quarterly_income_stmt")
        quarterly_balance_sheet.to_excel(writer, sheet_name=f"{ticker}_quarterly_balance_sheet")
        quarterly_cashflow.to_excel(writer, sheet_name=f"{ticker}_quarterly_cashflow")

        return 0

# Safely load raw data
def load_data():

    # Stock parameters
    ticker = "AAPL"
    period = "3y"
    interval = "1d"

    # Get stock data
    yfinance_data_to_excel(ticker, period, interval)

    # File paramaters
    file = "data/raw.xlsx"
    sheet_name = f"{ticker}_OHCL"


    # Load data from excel file
    if file.endswith(".xlsx"):
        data = pd.read_excel(file, sheet_name= sheet_name, index_col=0, parse_dates=True)
    elif file.endswith(".csv"):
        data = pd.read_csv(file)
    else:
        raise ValueError("File format not supported. Please use .xlsx or .csv files.")

    return data

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

#test features and targets into x and y respectively
def split_features_target(data, target_col):

    """test data into features (X) and target (y)."""
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


"""test data chronologically into a training set and a remainder (X_test, y_test)."""
def splitting_data(data, target_col, timesteps=10):

    """"Split data chronologically into a training set and a remainder (X_split, y_split)."""
    X, y = split_features_target(data, target_col)

    # Time-based split (no shuffling!)
    train_size = int(len(X) * 0.85)  # 80% for training
    
    X_train = X[:train_size]
    X_test = X[train_size:]
    
    y_train = y[:train_size]
    y_test = y[train_size:]

    #Store column names BEFORE transformation
    feature_columns_train = X_train.columns.tolist()

    # Transform data
    X_train, min_max_scaler = min_max_scale(X_train, feature_columns_train)

    #save scalers
    save_scaler_data(min_max_scaler)

    X_train = pd.DataFrame(X_train, columns = feature_columns_train)
    X_test = pd.DataFrame(min_max_scaler.transform(X_test), columns = feature_columns_train)

    # Align features
    X_train = align_features(X_train, feature_columns_train)
    X_test = align_features(X_test, feature_columns_train)

    # Create LSTM input
    X_train = create_lstm_input(X_train, feature_columns_train)
    X_test = create_lstm_input(X_test, feature_columns_train)

    # Step 3: Align targets
    y_train = y_train[timesteps:]
    y_test  = y_test[timesteps:]


    # Align features again after LSTM transformation
    check_feature_alignment(X_test, X_train)

    # Return traininig results 
    data ={
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_columns_train": feature_columns_train,
        "Close_series": data['Close'].values
    }
    
    return data

def split_train_val(X, y, val_frac=0.2):
    val_size = int(len(X) * val_frac)
    return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]
