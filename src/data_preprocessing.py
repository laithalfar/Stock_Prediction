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
    data['SMA_ratio'] = data['SMA_50'] / data['SMA_200']  # Fixed: lowercase to match features list
    

    # Calculate daily returns 
    data['Daily_Return'] = data['Close'].pct_change()
    
    
    # --- EMA (Exponential Moving Average) ---
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    # --- MACD (Moving Average Convergence Divergence) Histogram---
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

   
    # Normalized
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
    data["MA_Spread_pct"] = (data["SMA_50"] - data["SMA_200"]) / data["SMA_200"]

    #Next day return
    data['Next_Day_Return'] = data['Close'].pct_change().shift(-1)  # Tomorrow's return
    data['Rel_Volume_20'] = data['Volume'] / data['Volume'].rolling(20).mean()

    # --- Select Features + Target ---                                                                                                                                                                                                                                                  
    # Include all features AND the target column (Next_Day_Return)
    # Note: 'Close' deliberately excluded to prevent leakage
    features_and_target = ['SMA_ratio', 'Daily_Return','MACD_Histogram_Norm', 'RSI', 'BB_pct', 
                           'Volatility_30', 'Support_Dist_pct', 'Resistance_Dist_pct', 'Body_pct', 
                           'MA_Spread_pct', 'GoldenCross', 'DeathCross', 'Trend_Regime', 
                           'Rel_Volume_20', 'Range_Pct', 'Next_Day_Return']

    # Only include columns that exist in the data
    available_cols = [col for col in features_and_target if col in data.columns]
    df_clean = data[available_cols].dropna()  # Drop rows with NaN from rolling windows
    
    # Return full dataframe (features + target) and Close series for later use
    return df_clean, data["Close"]

# Standardize selected columns to mean=0, std=1.
def standard_scale(data_train, data_val, data_test, columns= None):
    

    """
    Standardize selected columns to mean=0, std=1.

    Parameters
    ----------
    data_train : pd.DataFrame
        Training data to fit the scaler to.
    data_val : pd.DataFrame
        Validation data to transform.
    data_test : pd.DataFrame
        Test data to transform.
    columns : list of str, optional
        Columns to standardize. If None, all numeric columns are standardized.

    Returns
    -------
    [nd.array]
        Standardized data.
    """

    data_train = data_train.copy()
    data_val = data_val.copy()
    data_test = data_test.copy()
    
    if columns is None:
        columns = data_train.select_dtypes(include=np.number).columns.tolist()
    
    scaler = StandardScaler()
    X_scaler = scaler.fit(data_train[columns])
    # Transform each split
    data_train_scaled = pd.DataFrame(
        scaler.transform(data_train[columns]),
        index=data_train.index,
        columns=columns,
    )
    data_val_scaled = pd.DataFrame(
        scaler.transform(data_val[columns]),
        index=data_val.index,
        columns=columns,
    )
    data_test_scaled = pd.DataFrame(
        scaler.transform(data_test[columns]),
        index=data_test.index,
        columns=columns,
    )

     # Put scaled columns back into the original frames
    # data_train.loc[:, columns] = data_train_scaled
    # data_val.loc[:, columns] = data_val_scaled
    # data_test.loc[:, columns] = data_test_scaled


    return data_train_scaled, data_val_scaled, data_test_scaled, X_scaler # return data

    
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

    data = data[feature_columns].values # get 2d Matrix with values and index
    X = []
    
    # Loop over the data skipping over the first timestep
    # because you will be subtracting the timestep value from the index.
    for i in range(timesteps, len(data)): 
        X.append(data[i-timesteps:i]) # Here the data appeneded is 2D which means X would be a group of 2D matrices 
        #which would make it a 3D matrix
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

    X = data.drop(columns=[target_col]) # keep in mind a dataframe is like a 2D Array but with for labels with columns (this does not mean 
    # there is no integer indexing for rows. It just means there are string labels as well now for columns)
    y = data[target_col] # Keep in mind a series is like a 1D Array but with labels for the one column as well

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

    # Add missing columns
    for col in train_columns: 
        if col not in X_df.columns:
            X_df[col] = 0

    # Remove extra columns
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

# Walk forward validation function
def walk_forward_validation(X, y, train_window=252, test_window=21):
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


# Test data chronologically into a training set and a remainder (X_test, y_test).
def preprocess(data, target_col, close, timesteps = 10):

    # Step 1: Split features and target
    X, y = split_features_target(data, target_col)
    
    # 2. Store column names BEFORE transformation
    feature_columns_X = X.columns.tolist() 
    
    # 3. Set variable list
    X_te_scaled_list, y_te_scaled_list, X_val_scaled_list, y_val_scaled_list, y_tr_scaled_list, X_tr_scaled_list, close_te_list, fold_list, X_scaler_list, y_scaler_list = [], [], [], [], [], [], [], [], [], []

    # 4. Time-based split (no shuffling!)
    try: 
        # Fold is a counter for each walk-forward iteration
        # Each iteration trains on a rolling window and tests on the subsequent window
        # This simulates real-world sequential prediction and the iteration is done using the enumerate function
        for fold, (X_tr, y_tr, X_te, y_te, end_train, end_test) in enumerate(
            walk_forward_validation(X, y, train_window=252, test_window=21)
        ):
            
            # 5. Split training window into train+val (time-ordered)
            X_tr, y_tr, X_val, y_val = split_train_val(X_tr, y_tr, val_frac = 0.2)

            # 6. Check feature alignment
            check_feature_alignment(X_te, X_tr)
            check_feature_alignment(X_val, X_tr)

            # 7. Scale data
            X_tr_scaled, X_val_scaled, X_te_scaled, X_scaler = standard_scale(X_tr, X_val, X_te)

            #    - y: convert Series -> DataFrame so standard_scale works
            y_tr_df = y_tr.to_frame()
            y_val_df = y_val.to_frame()
            y_te_df = y_te.to_frame()


            y_tr_scaled, y_val_scaled, y_te_scaled, y_scaler = standard_scale(y_tr_df, y_val_df, y_te_df)

            # Back to Series (optional; arrays are also fine downstream)
            y_tr_scaled = y_tr_scaled.iloc[:, 0]
            y_val_scaled = y_val_scaled.iloc[:, 0]
            y_te_scaled  = y_te_scaled.iloc[:, 0]

            # 10. Align X features
            X_tr_scaled = align_features(X_tr_scaled, feature_columns_X)
            X_val_scaled = align_features(X_val_scaled, feature_columns_X)
            X_te_scaled = align_features(X_te_scaled, feature_columns_X)

            # 11. Create LSTM input
            X_tr_scaled = create_3D_input(X_tr_scaled, feature_columns_X)
            X_val_scaled = create_3D_input(X_val_scaled, feature_columns_X)
            X_te_scaled = create_3D_input(X_te_scaled, feature_columns_X)

            # 12. Align targets
            y_tr_scaled =  y_tr_scaled[timesteps:]
            y_val_scaled =  y_val_scaled[timesteps:]
            y_te_scaled  =  y_te_scaled[timesteps:]

            # Align features again after LSTM transformation
            # 13. Assert alignment
            assert X_tr_scaled.shape[0] == len(y_tr_scaled), \
                 f"Mismatch: X_train has {X_tr_scaled.shape[0]} samples but y_train has {len(y_tr_scaled)} targets"
            assert X_val_scaled.shape[0] == len(y_val_scaled), \
                 f"Mismatch: X_val has {X_val_scaled.shape[0]} samples but y_val has {len(y_val_scaled)} targets"
            assert X_te_scaled.shape[0] == len(y_te_scaled), \
                 f"Mismatch: X_test has {X_te_scaled.shape[0]} samples but y_test has {len(y_te_scaled)} targets"

            # Save scalers
            #Save_scaler_data(Standard_scaler)

            close_te = close[end_train : end_test+1]
            print("close_te: ", close_te)

            # 14. Store fold data in list 
            X_tr_scaled_list.append(X_tr_scaled)
            y_tr_scaled_list.append(y_tr_scaled)
            X_val_scaled_list.append(X_val_scaled)
            y_val_scaled_list.append(y_val_scaled)
            X_te_scaled_list.append(X_te_scaled)
            y_te_scaled_list.append(y_te_scaled)
            X_scaler_list.append(X_scaler)
            y_scaler_list.append(y_scaler)
            close_te_list.append(close_te)
            fold_list.append(fold)


    except Exception as e:
        print(f"[CRITICAL ERROR] Walk-forward training failed: {e}")
        raise

    # Return traininig results 
    returned_data = {
        "X_train": X_tr_scaled_list,
        "y_train": y_tr_scaled_list,
        "X_val": X_val_scaled_list,
        "y_val": y_val_scaled_list,
        "X_test": X_te_scaled_list,
        "y_test": y_te_scaled_list,
        "close_te_list": close_te_list,
        "folds": fold_list,
        "y_scaler_list": y_scaler_list,
        "X_scaler_list": X_scaler_list,
        "feature_columns_X": feature_columns_X
    }
    
    return returned_data
