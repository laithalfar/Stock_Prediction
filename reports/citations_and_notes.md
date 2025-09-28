# [x] Calculating moving averages as https://www.investopedia.com/terms/m/movingaverage.asp 

# [x] Calculating daily returns aka intraday return https://www.investopedia.com/terms/i/intraday-return.asp

# Why daily returns are useful for analyzing stock performance : 
# - Converts prices (which trend upward over time) into returns (which oscillate around 0).
# - Allows for calculation of cumulative return.

# [ ] Take in mind stock split and reverse stock-split https://www.investopedia.com/terms/c/closingprice.asp

# [x] Justifying RSI, MACD and EMA calculation https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjmssPy8N-PAxWMRaQEHRYiJpEQFnoECBgQAQ&url=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs41599-024-02807-x&usg=AOvVaw1n25Mawbi2k6TSfht5Zp8P&opi=89978449

# Short-term signals (RSI, EMA_12, Daily_Return)

# Mid-term trend (EMA_26, EMA_50, MACD) 

# Long-term bias (EMA_200, volatility) 

# Liquidity/interest (volume features)

# [x] Looking into VWMAs https://www.fisdom.com/how-to-calculate-closing-stock-price/ # Excluded because too highly correlated with most features

# [x] 30 days moving standard deviation of ‘Close’ (Close_MSTD) 
# [x] One day lagged feature of ‘Close’ (Close_Lag) 
# [x] One day lagged feature of ‘Open’ (Open_Lag) 
# [x] One day lagged feature of ‘High’ (High_Lag) 
# [x] One day lagged feature of ‘Low’ (Low_Lag) 
# [x] One day lagged feature of ‘Volume’ (Volume_Lag) 
# Excluded because too highly correlated with Close, open, high, low, volume

# file:///C:/Users/laith/Downloads/3501-10793-1-PB.pdf

# [x] Bollinger Bands (AI Overview)

 # [x] Identifying price points where a stock tends to find buying (support) or selling (resistance) pressure can help predict price boundaries

 # [x] Daily Volatility (Daily_Range, Range_Pct)

 # Creating plots for EDA to determine trends and patterns in the data:
 
    - Line plots for daily closing prices and moving averages.
    - Histograms for daily returns and volatility.
    - Feature correlation heat map.

# Excluded features because too highly correlated with most features:
    - EMA_50 
    - EMA_200
    - Cumulative_returns
    - # One day lag for Close, open, high, low, volume


# Option 3: Hybrid approach

# Train on returns (stationary) but convert back to prices for prediction
# Predict next day's return, then: predicted_close = current_close * (1 + predicted_return)


## Why the split_train_val() is still needed 
<!-- 

Walk-forward has two layers of splitting:

Outer split (generator):

Produces a rolling training window (X_tr_full, y_tr_full) and the next unseen window (X_te, y_te) for testing.

This ensures chronological integrity.

Inner split (split_train_val):

Takes the training window (X_tr_full) and holds out a slice from its tail as validation.

Validation is used only for callbacks (early stopping, checkpointing).

Without this, your callbacks would use the test window as “val_loss”, which leaks information.

So:

The concatenation makes sure you’re working with the entire time series in chronological order.

The outer generator enforces rolling train/test.

The inner split gives you a proper validation slice inside the training window.

“Rolling training window” is just a time-series way of saying:

“Take a fixed chunk of the past, train on it, then roll the window forward and repeat.”

Imagine you have 1,000 trading days of stock data.

Window size: say 252 days (≈ 1 trading year).

Step size / test size: say 21 days (≈ 1 trading month).

Now, instead of training once and testing once, you do this:

Fold 1: Train on days 1–252 → test on days 253–273.

Fold 2: Train on days 22–273 → test on days 274–294.

Fold 3: Train on days 43–294 → test on days 295–315.
… and so on.

Each time the “training window” slides (or rolls) forward in time, always using the most recent history to predict the immediate future.

Normally, if you loop over your generator:

for fold_data in walk_forward_validation(X, y, train_window=252, test_window=21):
    ...


then fold_data would be a tuple like (X_tr_full, y_tr_full, X_te, y_te) but you wouldn’t know which fold you’re on unless you manually tracked it.

By wrapping it in enumerate:

for fold, (X_tr_full, y_tr_full, X_te, y_te) in enumerate(
    walk_forward_validation(X, y, train_window=252, test_window=21)
):


you get two things on each iteration:

fold → the loop index (starting at 0 by default).

(X_tr_full, y_tr_full, X_te, y_te) → the actual values from the generator.
 -->


## Future enhancements:

# - Account for changing raw data