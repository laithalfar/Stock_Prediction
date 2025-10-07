# 1. Executive Summary  (~250–350 words)
## 1.1 Introduction
The goal of this project was to develop and evaluate neural network models for a time-series forecasting task. Among options such as energy consumption and air quality prediction, I chose stock closing-price forecasting for its complexity and practical relevance.

Three models were implemented and optimized through hyperparameter tuning and walk-forward validation: a Long Short-Term Memory network (LSTM), a Recurrent Neural Network (*RNN*), and a hybrid Convolutional Neural Network–Gated Recurrent Unit (*CNN-GRU*). The CNN-GRU was selected as the custom architecture for its potential to capture both local temporal patterns (via *CNN* layers) and long-term dependencies (via *GRU*s).

Evaluation using metrics such as *RMSE*, *MAE*, and *R²* showed that the LSTM achieved the most accurate and consistent performance across folds (*average RMSE* ≈ 1.20, *MAE* ≈ 0.88), outperforming both the RNN and CNN-GRU. This suggests that the dataset’s predictive structure relies on longer-term temporal dependencies rather than short local patterns.

Final Recommendation:
The LSTM model should be considered the preferred architecture for forecasting stock closing prices in similar time-series contexts, given its balance of accuracy and stability across temporal folds.


# 2. Problem Definition
## 2.1 Project Objective

This project aims to develop and evaluate neural network architectures for a time-series forecasting task focused on stock price prediction. The objective is to forecast the daily closing price of Apple Inc. (AAPL) using engineered financial indicators and deep learning models capable of capturing both short-term and long-term dependencies.
Stock data presents complex temporal dynamics and non-stationarity, making feature engineering crucial. Features are designed to extract relative, scale-independent patterns that describe momentum, volatility, and trend behavior rather than raw price levels. These include moving averages, oscillators, and ratio-based indicators, all intended to help models generalize across varying market regimes.

## 2.2 Dataset Description

The dataset is Apple’s historical daily stock data obtained via the yfinance library, spanning the past three years. It contains the standard OHLCV structure —
- *Open*: The first traded price of the asset when the market opens for that period (day, hour, etc.). It represents where traders were willing to begin trading after the previous close — often used as a psychological “starting point” for that session’s sentiment.

- *High*: The maximum price reached during the trading period. It reflects the highest level of buyer enthusiasm or price acceptance before sellers pushed back.

- *Low*: The minimum price reached during the trading period. It shows the lowest point of selling pressure before buyers stepped in.

- *Close*: The last traded price when the market closed for that period. It’s often considered the most important of the four, because it reflects the final consensus value of the asset at day’s end and is used in most technical indicators (moving averages, RSI, Bollinger Bands, etc.).


- *Volume* : The total number of shares (or contracts) traded during the period. It measures market activity and participation — higher volume indicates stronger conviction behind price moves, while low volume suggests apathy or uncertainty.

Some basic statistics for the features we have:
Key Figures & Snapshot

Current Price (latest close): ~ USD 256.48

Intraday Range: High ≈ USD 257.38, Low ≈ USD 255.45

Open (today): USD 256.84

Volume: 31.9M


## 2.3 Success Metrics

The Mean Absolute Error (MAE) is the primary evaluation metric, as it robustly measures the average deviation between predicted and actual closing prices. MAE offers a stable, interpretable metric for comparing model performance across architectures while mitigating the effects of outliers inherent in financial data.

# 3. Exploratory Data Analysis (EDA) and Preprocessing  (~500–600 words)
## 3.1 Data Exploration & Visualization
Embed 3–5 plots using Markdown:
`![Closing Prices](plots/closing_prices.png)`
`![Correlation Heatmap](plots/correlation.png)`
Include a few lines of interpretation under each.

## 3.2 Data Cleaning & Feature Engineering
Explain missing value handling, scaling, feature creation, and LSTM sequence setup.


SMA_Ratio: Ratio of the 50-day to 200-day simple moving average, indicating medium-term momentum strength.

Daily_Returns: Day-to-day percentage change in the closing price.

MACD_Histogram_Norm: Normalized MACD histogram relative to EMA-26, capturing short-term momentum acceleration or deceleration.

RSI: Relative Strength Index identifying overbought (>70) and oversold (<30) conditions.

Volatility_30: 30-day rolling standard deviation of the closing price normalized by the current close.

BB_pct: Position of the current price within Bollinger Bands, identifying price extremes or mean-reversion signals.

Support_Dist_pct / Resistance_Dist_pct: Relative distance from the lower and upper Bollinger Bands, respectively.

Range_pct: Intraday volatility range relative to price.

Body_pct: Percentage change between open and close, representing directional strength.

GoldenCross / DeathCross: Moving average crossover indicators marking potential bullish or bearish regime shifts.

Trend_Regime: Market phase classification derived from moving average relationships.

MA_Spread_pct: Percentage spread between short- and long-term moving averages, quantifying trend strength.

Rel_Volume_20: Relative trading volume compared to the 20-day average, reflecting market participation intensity.

Data preprocessing involved loading from yfinance, removing duplicates and NaN values, feature extraction, dataset splitting, scaling, feature alignment between train/test sets, and generating 3D input sequences for neural networks.

## 3.3 Data Splitting
Describe how you split train/validation/test sets (with ratios).

# 4. Model Development and Training  (~600–800 words)
For each model (RNN, LSTM, Custom):
## 4.x.1 Model Architecture
Paste your `model.summary()` inside triple backticks for readability.
## 4.x.2 Hyperparameters
List optimizer, loss, epochs, batch size, etc.
## 4.x.3 Training Performance
Embed training/validation loss plots and comment briefly.

# 5. Model Evaluation and Comparison  (~400–500 words)
## 5.1 Test Performance
Add a Markdown table comparing all models:

| Model | MAE | RMSE | R² |
|--------|-----|------|----|
| RNN | ... | ... | ... |
| LSTM | ... | ... | ... |
| CNN-GRU | ... | ... | ... |

## 5.2 Interpretation
Discuss which model wins and hypothesize why.
## 5.3 Best Model Visualization
Include a prediction-vs-true plot.

# 6. Conclusion  (~250–300 words)
## 6.1 Summary of Findings
Restate your final conclusion clearly.
## 6.2 Limitations
Be honest about data or modeling constraints.
## 6.3 Future Work
Give 2–3 actionable next steps.

# 7. Appendix
## 7.1 Environment
List key libraries and versions in a Markdown code block.
## 7.2 Link to Code
Provide the GitHub or local repo path.


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
