import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf

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

#remove outliers
def remove_outliers_iqr(data, columns, k=1.5):
    """
    Remove rows with outliers in given columns using the IQR rule.
    k=1.5 -> standard, can increase for less aggressive removal.
    """
    clean_df = data.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(clean_df[col]):
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - k * IQR
            upper = Q3 + k * IQR
            clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]
    return clean_df

#  Clean data from duplicates and NaNs
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Fix missing values, duplicates, datatypes."""
    data = data.drop_duplicates()
    data = data.fillna(method="ffill")  # forward fill missing values
    return data


# Process the data
def feature_engineering(data):

    # Calculate moving averages, and cumulative returns
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['SMA_500'] = data['Close'].rolling(window=500).mean()

    # Calculate daily returns and cumulative daily returns
    data['Daily_Return'] = data['Close'].pct_change()
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
    

    # --- MACD (Moving Average Convergence Divergence) ---
    # --- Signal line ---
    # --- EMA (Exponential Moving Average) ---
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

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

    # --- Volume Weighted Moving Average (VWMA's) ---
    data['VWMA_20'] = (data['Close'] * data['Volume']).rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()

    # 30-day rolling standard deviation of 'Close' aka volatility 
    data['STD_30'] = data['Close'].rolling(window=30).std()

    # --- Lagged Features ---
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        data[f'{col}_Lag'] = data[col].shift(1)

    # Bollinger Bands parameters
    window = 20
    k = 2

    # --- Bollinger Bands ---
    data['BB_Middle'] = data['Close'].rolling(window=window).mean()
    data['BB_STD'] = data['Close'].rolling(window=window).std()
    data['BB_Upper'] = data['BB_Middle'] + k * data['BB_STD']
    data['BB_Lower'] = data['BB_Middle'] - k * data['BB_STD']
    data['BB_pct'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    # --- Remove outliers ---
    #data = remove_outliers_iqr(data, data.select_dtypes(include='number').columns.tolist())

    # --- Select Features ---
    features = ['SMA_50', 'SMA_200', 'SMA_500', 'Daily_Return', 'Cumulative_Return',
                'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', 'EMA_50', 'EMA_200',
                'Close_Lag', 'Open_Lag', 'High_Lag', 'Low_Lag', 'Volume_Lag', 'BB_pct', 'VWMA_20', 'STD_30', 'Close', 'Open', 'High', 'Low', 'Volume']

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

def one_hot_encode(df, columns=None):
    """
    One-hot encode categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
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
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df

#change to 3D for lstm input
def create_lstm_input(df, feature_columns, timesteps=10):
    """
    Convert dataframe to 3D array for LSTM:
    [samples, timesteps, features].
    
    Parameters:
    - df: pandas DataFrame, already scaled/cleaned
    - feature_columns: list of column names to use as features
    - timesteps: number of previous steps to include
    
    Returns:
    - X: np.array of shape [samples, timesteps, features]
    """
    data = df[feature_columns].values
    X = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
    X = np.array(X)
    return X

# Transform data
def data_transformation(data):
    """Transform data for model training."""

    #one hot encode categorical variables
    data = one_hot_encode(data)

     #scale data
    data, min_max_scaler = min_max_scale(data)

    return data, min_max_scaler
    
#split features and targets into x and y respectively
def split_features_target(df, target_col, timesteps=10):

    """Split data into features (X) and target (y)."""
    X = create_lstm_input(df, df.columns.drop(target_col), timesteps)
    y = df[target_col].values[timesteps:]
    return X, y


#splitting data into training, val and testing sets
def splitting_data(data, target_col, timesteps=10):
    """Split data into training and testing sets."""
    X, y = split_features_target(data, target_col, timesteps)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=432)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=432)

    return X_train, X_test, y_train, y_test, X_val, y_val

# align features to have the same index
def align_features(X_test, X_train_columns):
    """Ensure test data has the same features as training data."""

    # Add missing columns
    for col in X_train_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Remove extra columns
    for col in list(X_test.columns):
        if col not in X_train_columns:
            X_test = X_test.drop(col, axis=1)
    
    # Ensure columns are in the same order
    X_test = X_test[X_train_columns]
    
    return X_test