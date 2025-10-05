import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import joblib
import os
import sys


# Add project root to sys.path so imports work
sys.path.append(os.path.abspath(".."))

from config import SCALER_X_PATH

# Extract stock data from yfinance
def yfinance_data_to_excel(ticker, period, interval):
        
    """
    Saves data from yfinance to an excel file in different sheets for better viewing and analyses.
    
    Parameters
    ----------
    ticker : str
        The ticker symbol of the stock.
    period : str
        The time period for which you want to collect data.
    interval : str
        The interval at which you want to collect data.

    Returns
    -------
    int
        0 if successful, otherwise an error code.
    """

    # Get stock data
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

   

    """
    Safely load raw data from an excel file.

    Parameters
    ----------
    ticker : str
        The stock ticker you want to load data for.
    period : str
        The period for which you want to collect data.
    interval : str
        The interval at which you want to collect data.

    Returns
    -------
    pandas.DataFrame
        The loaded data.
    """

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
  
    """
    Clean data from duplicates and NaNs.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be cleaned.

    Returns
    -------
    pandas.DataFrame
        The cleaned data.
    """
    
    data = data.drop_duplicates()
    data = data.interpolate(method="linear")  # New syntax
    return data


# Process the data
def feature_engineering(data):

    
    """
    Calculate various features from the given data.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be processed.

    Returns
    -------
    pandas.DataFrame
        The processed data with the added features.
    """

    # Calculate moving averages, and cumulative returns
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['SMA_Ratio'] = data['SMA_50'] / data['SMA_200']
    

    # Calculate daily returns 
    data['Daily_Return'] = data['Close'].pct_change()
    
    
    # --- EMA (Exponential Moving Average) ---
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    # --- MACD (Moving Average Convergence Divergence) Histogram---
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

   
    #normalized
    data['MACD_Histogram_Norm'] = data['MACD_Histogram'] / data['EMA_26']


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
    data['Volatility_30'] = data['STD_30'] / data['Close']

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
    data['Support_Dist_pct'] = (data['Close'] - data['Rolling_Min_20']) / data['Close']
    data['Resistance_Dist_pct'] = (data['Rolling_Max_20'] - data['Close']) / data['Close']

    #Daily Volatility
    data["Daily_Range"] = data["High"] - data["Low"]
    data["Range_Pct"] = (data["High"] - data["Low"]) / data["Close"]

    #candlestick body (bearish vs bullish)
    data["Candlestick_Body"] = data["Close"] - data["Open"]
    data['Body_pct'] = (data['Close'] - data['Open']) / data['Open']


    #GoldenCross and DeathCross
    data["GoldenCross"] = ((data["SMA_50"] > data["SMA_200"]) & (data["SMA_50"].shift(1) <= data["SMA_200"].shift(1))).astype(int)
    data["DeathCross"] = ((data["SMA_50"] < data["SMA_200"]) & (data["SMA_50"].shift(1) >= data["SMA_200"].shift(1))).astype(int)
    data["Trend_Regime"] = (data["SMA_50"] > data["SMA_200"]).astype(int)

    #MA spread and regime
    data["MA_Spread_pct"] = data["SMA_50"] - data["SMA_200"]/data["SMA_200"]


    data['Next_Day_Return'] = data['Close'].pct_change().shift(-1)  # Tomorrow's return
    data['Rel_Volume_20'] = data['Volume'] / data['Volume'].rolling(20).mean()

    # --- Select Features ---
    features = ['SMA_ratio', 'Daily_Return','MACD_Histogram_Norm', 'RSI', 'BB_pct', 'Volatility_30', 'Support_Dist_pct', 'Resistance_Dist_pct', 'Body_pct', 'MA_Spread_pct', 'GoldenCross', 'DeathCross', 'Trend_Regime', 'Close', 'Rel_Volume_20', 'Range_Pct', 'Next_Day_Return']

    # Only include features that exist in the data
    available_features = [f for f in features if f in data.columns]
    X = data[available_features].dropna()  # Drop rows with NaN from rolling windows
    
    return X

# Standardize selected columns to mean=0, std=1.
def Standard_scale(X_train, columns=None):
    
    """
    Standardize selected columns to mean=0, std=1.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The dataframe to standardize.
    columns : list of str, optional
        The columns to standardize. If None, use all numeric columns.

    Returns
    -------
    X_train : pandas.DataFrame
        The dataframe with standardized columns.
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used to standardize the columns. Use this to standardize the test set.
    """
    X_train = X_train.copy()
    
    if columns is None:
        columns = X_train.select_dtypes(include=np.number).columns.tolist()
    
    scaler = StandardScaler()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    return X_train, scaler  # return scaler to apply to test set

    
# Change to 3D for lstm input
def create_3D_input(data, feature_columns, timesteps=10):

    """
    Convert dataframe to 3D array for LSTM/RNN/CNN_GRU:
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

# Split a DataFrame into features (X) and target (y).
def split_features_target(data, target_col):

    
    """
    Split a DataFrame into features (X) and target (y).
    
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to split.
    target_col : str
        The column name of the target variable.
    
    Returns
    -------
    X : pandas.DataFrame
        The features.
    y : pandas.Series
        The target variable.
    """

    X = data.drop(columns=[target_col])
    y = data[target_col]

    return X, y

# Make sure test and train data have same features
def align_features(X_df, train_columns):
    
    """
    Align the features of X_df with the columns of train_columns.
    
    Parameters
    ----------
    X_df : pandas.DataFrame
        The DataFrame to align.
    train_columns : list
        The columns to align with.
    
    Returns
    -------
    pandas.DataFrame
        The aligned DataFrame.
    """

    for col in train_columns:
        if col not in X_df.columns:
            X_df[col] = 0
    for col in list(X_df.columns):
        if col not in train_columns:
            X_df = X_df.drop(col, axis=1)
    return X_df[train_columns]


# Check feature alignment worked well
def check_feature_alignment(X_test, X_train):
   
    """Check that the features of X_test and X_train are aligned.
    Raise a ValueError if there is a mismatch. Returns X_test if there is no mismatch."""

    if list(X_test.columns) != list(X_train.columns):
        raise ValueError(
            f"Feature mismatch!\n"
            f"Train columns: {list(X_train.columns)}\n"
            f"Test columns: {list(X_test.columns)}"
        )
    return X_test


# Save scalers
def save_scaler_data(Standard_scaler):
    
   
    """
    Save the StandardScaler to a file.

    Parameters
    ----------
    StandardScaler : StandardScaler
        The StandardScaler to save.

    Returns
    -------
    None
    """
     # 🔽 Save scalers here
    joblib.dump(Standard_scaler, SCALER_X_PATH)


# Test data chronologically into a training set and a remainder (X_test, y_test).
def splitting_data(data, target_col, timesteps=10):

    
    """
    Split data into training and testing sets using a time-based split for lstm.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the features and target.
    target_col : str
        Name of the target column.
    timesteps : int, optional
        Number of timesteps to use for the LSTM model (default: 10).

    Returns
    -------
    dict
        Dictionary containing the training and testing data, as well as the feature columns and Close series.
    """
    # Step 1: Split features and target
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
    X_train, Standard_scaler = Standard_scale(X_train, feature_columns_train)
    X_train = pd.DataFrame(X_train, columns=feature_columns_train, index=X_train.index)
    X_test = pd.DataFrame(Standard_scaler.transform(X_test), columns=feature_columns_train, index=X_test.index)

    # Step 2: Check feature alignment
    check_feature_alignment(X_test, X_train)

    #save scalers
    save_scaler_data(Standard_scaler)

    # Align features
    X_train = align_features(X_train, feature_columns_train)
    X_test = align_features(X_test, feature_columns_train)

    # Create LSTM input
    X_train = create_3D_input(X_train, feature_columns_train)
    X_test = create_3D_input(X_test, feature_columns_train)

    # Step 3: Align targets
    y_train = y_train[timesteps:]
    y_test  = y_test[timesteps:]


    # Align features again after LSTM transformation

    # 7. Assert alignment
    assert X_train.shape[0] == len(y_train), \
        f"Mismatch: X_train has {X_train.shape[0]} samples but y_train has {len(y_train)} targets"
    assert X_test.shape[0] == len(y_test), \
        f"Mismatch: X_test has {X_test.shape[0]} samples but y_test has {len(y_test)} targets"


    # Return traininig results 
    returned_data ={
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_columns_train": feature_columns_train,
        "Close_series": data['Close'].values
    }
    
    return returned_data

def split_train_val(X, y, val_frac=0.2):
    
    """
    Split X and y into training and validation sets.

    Parameters
    ----------
    X : array-like
        Feature data
    y : array-like
        Target data
    val_frac : float, optional
        Fraction of data to use for validation (default: 0.2)

    Returns
    -------
    X_train, y_train, X_val, y_val
        Split data into training and validation sets
    """

    val_size = int(len(X) * val_frac)
    return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]
