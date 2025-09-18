import pandas as pd

#load data from excel file
load_data = pd.read_excel("data/raw.xlsx", sheet_name="AAPL_OHCL")


 #Bollinger Bands, and the Relative Strength Index (RSI), convergence divergence moving average (MACD), 

 # Identifying price points where a stock tends to find buying (support) or selling (resistance) pressure can help predict price boundaries
def preprocess_data(data):

    # Calculate moving averages, and cumulative returns
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['SMA_500'] = data['Close'].rolling(window=500).mean()

    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()

    # Calculate cumulative returns using daily returns
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()

    # --- EMA (Exponential Moving Average) ---
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    # --- MACD (Moving Average Convergence Divergence) ---
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # --- RSI (Relative Strength Index) ---
    window_length = 14
    delta = data["Close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()

    # Use exponential moving average for smoother RSI
    avg_gain = avg_gain.ewm(com=window_length-1, adjust=False).mean()
    avg_loss = avg_loss.ewm(com=window_length-1, adjust=False).mean()

    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # --- EMA (Exponential Moving Average) ---
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()

    return data

# align features to have the same index
def align_features(data):
    return data