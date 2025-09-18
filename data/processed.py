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

    return data

# align features to have the same index
def align_features(data):
    return data