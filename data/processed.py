import pandas as pd

def remove_outliers_iqr(df, columns, k=1.5):
    """
    Remove rows with outliers in given columns using the IQR rule.
    k=1.5 -> standard, can increase for less aggressive removal.
    """
    clean_df = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(clean_df[col]):
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - k * IQR
            upper = Q3 + k * IQR
            clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]
    return clean_df


# Load data from excel file
load_data = pd.read_excel("data/raw.xlsx", sheet_name="AAPL_OHCL")

# Process the data
def preprocess_data(data):

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
    data = remove_outliers_iqr(data, data.select_dtypes(include='number').columns.tolist())

    # --- Select Features ---
    features = ['SMA_50', 'SMA_200', 'SMA_500', 'Daily_Return', 'Cumulative_Return',
                'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI', 'EMA_50', 'EMA_200',
                'Close_Lag', 'Open_Lag', 'High_Lag', 'Low_Lag', 'Volume_Lag', 'BB_pct', 'VWMA_20', 'STD_30']

    # Only include features that exist in the data
    available_features = [f for f in features if f in data.columns]
    X = data[available_features].dropna()  # Drop rows with NaN from rolling windows
    
    return X

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