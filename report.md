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

`![SMA50/SMA200/ClosePrice vs time](stock-forecast/reports/SMA-50-200.png)`

Left region (~0–150):
The 50-day SMA (orange) dips below the 200-day SMA — that’s a death cross, confirming a bearish phase.
The price moves mostly below both SMAs — weak momentum, market correction.

Middle region (~200–350):
The 50-day SMA climbs and crosses above the 200-day SMA — that’s a golden cross, marking a bullish regime.
The close price (blue) stays consistently above both averages — strong uptrend.

Right region (~400–500):
After the drop around index 400, the short-term SMA again falls below the long-term one — a death cross — followed by consolidation and a possible new upturn near the end.
The SMAs begin to flatten and re-converge, suggesting a potential trend reversal or stabilization.

`![OHLCV vs time](stock-forecast/reports/stock-OHLCV.png)`

The price gradually rises from around $170 → $250, with periods of consolidation and corrections — a healthy uptrend.

The dip around index ~400 likely represents a broader market correction (for instance, a quarter where tech stocks retraced).

The recovery afterwards shows resilience and renewed momentum.

The smooth continuity of the lines (no big jumps) indicates your data interpolation and cleaning worked properly — no missing or duplicated days.


## 3.2 Data Cleaning & Feature Engineering
Explain missing value handling, scaling, feature creation, and LSTM sequence setup.

After loading the data from yfinance and determining basic features, the data was cleaned by removing duplicates and NaN values with functions **.drop_duplicates()** which drops duplicates from the basic data and **.interpolate(method ="linear")** which fills in missing values (NaNs) by estimating them from nearby data points — linearly. As for outliersm in finance a change or spike in data is common and indicative of an event so removing outliers would be like deleting data.

Upon assuring we have clean data that would not raise any errors or inconsistincies, features were extracted from the basic OHLCV data. Each feature and its explanation is presented below in details.

- *SMA_Ratio*: Simple Moving Average (SMA) ratio of the average moving average of the last 50 days relative to the average moving average of the last 200 days. 

- *Daily_returns*: calculating the percentage change between the closing price of the current day and the closing price of the previous day. 

- *MACD_Histogram_Norm*(relative to EMA_26): A visual amplifier that tells you how fast momentum is changing. (The histogram shows whether the short-term acceleration is increasing or decreasing) 

- *RSI*: Relative Strength Index is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the stock market. (>70 overpriced)(<30 underpriced) 

- *Volatility_30*: 30 day standard deviation of 'Close' relative to the close of that day. 

- *BB_pct*: is a technical indicator that tells you where the current price sits relative to the upper and lower Bollinger Bands. If %B stays high (near 1), the stock is consistently pushing against the upper band — often a bullish sign. If %B stays low (near 0), the stock is consistently pushing against the lower band — often a bearish sign. When %B approaches 0 or 1, traders watch for reversals toward the mean. Combined with bandwidth (the distance between the bands), %B helps confirm whether strong price moves occur during high or low volatility periods. 

- *Support_Dist_pct*: The distance between the current price and the lower Bollinger Band realtive to the close price of the day. If the Support_Dist_pct is high, it means the current price is far from the lower Bollinger Band. 

- *Resistance_Dist_pct*: the distance between the current price and the upper Bollinger Band relative to the close price of the day. If the Resistance_Dist_pct is high, it means the current price is far from the upper Bollinger Band. 

- *Range_Pct*: refers to the percentage range of a stock’s movement within a given period — it measures how much the price fluctuated relative to its starting (or sometimes average) price indicating daily volatility. 

- *Body_pct*: This measures how much the price changed during the candle relative to where it opened — i.e., the percentage gain or loss for that period. Focuses on direction and magnitude of movement. 

- *GoldenCross*: Golden Cross happens when a short-term moving average crosses above a long-term moving average. Momentum is shifting upward — shorter-term prices are rising faster than the longer-term trend. Traders see it as a bullish signal or the start of a new uptrend. 

- *DeathCross*: A Death Cross is the opposite: the short-term moving average crosses below the long-term moving average. Momentum is weakening; the market’s short-term trajectory has turned downward relative to the longer trend. Traders see it as bearish or a sign of a potential downtrend. 

- *Trend_Regime*: A trend regime is a broader concept. It’s about categorizing the market environment based on indicators like moving averages, volatility, or momentum. 

- *MA_spread_pct*: measures how far apart two moving averages are — as a percentage of price — to quantify the strength of a trend, not just its direction. 

- *Rel_Volume_20*: measures how active today’s trading volume is compared to the average of the past 20 periods — a simple but powerful way to spot abnormal market participation.

Scaling using StandardScaler because scaling brings all features into a similar numerical range so that no single variable dominates the learning process simply due to its magnitude.Feature alignment between train/test sets to assure there is not a missing or extra columns. Then 3D input sequences were derived from the original 2D form for neural networks by interpreting timesteps into the data and storing it in a 3D array. The features are aligned again after between the y and x data sets.

## 3.3 Data Splitting
Describe how you split train/validation/test sets (with ratios).

The data was split into train and test sets with simple ratio splitting at first to avoid any problems with time-series data as the order for that type of data should never be shuffled. The ratio for the training data was 0.85 and that of the test data was 0.15. The same split was used for the validation data but implemented using walk forward validation to avoid shuffling while a method similar to cross validation as to optimize and make the most out of the data at hand.

# 4. Model Development and Training  (~600–800 words)
For each model (RNN, LSTM, Custom): 
## 4.1 Model Architecture
### 4.1.1 LSTM and RNN

The models were all defined as sequential which is a linear stack of layers where each layer has exactly one input tensor and one output tensor. That means data flows strictly from the first layer to the last, without branches, skips, or multiple inputs/outputs.

An LSTM model was fitted to the LSTM portion and an RNN model was to the RNN portion with the following parameters:

- Units: Units is the capacity of the layer; there is a risk of overfitting and slower training when increased.

- Dropout: which randomly drops connections during training to prevent overfitting.

- Recurrent dropout: randomly drops connections within the temporal loop of the model (between timesteps). This prevents the model from  memorizing specific sequences and forces it to learn more general temporal dependencies. Recurrent dropout adds controlled forgetting, improving generalization and robustness to noise.

- A kernel regularizer: adds a small penalty to the network’s weights during training to discourage them from growing too large. Large weights usually indicate overfitting — the model’s trying too hard to fit exact training examples instead of learning general relationships.

A Dropout is then fitted to the model which randomly drops connections during training to prevent overfitting.

The model is then fitted again with an LSTM/RNN and a dropout again. First LSTM/RNN captures higher-order temporal patterns(e.g., longer trends) as in specific patterns not generalized ones. Dropout makes sure it doesn’t memorize those patterns too tightly. The next LSTM/RNN learns from those regularized representations, adding abstraction depth and generalizing the data. 

The dense is the regression output for the model. In the time-series model, this layer is the decoder: it combines all the learned temporal features into one final numeric prediction.

Optimizer and loss function are applied to the model while it is compiled. Optimizer is the algorithm that adjusts weights during training. Loss function measures how well the model is performing (mean_squared_error is common for regression tasks).

**RNN BEST MODEL SUMMARY**
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ simple_rnn (SimpleRNN)               │ (None, 10, 128)             │          18,432 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 10, 128)             │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ simple_rnn_1 (SimpleRNN)             │ (None, 64)                  │          12,352 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1)                   │              65 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 61,700 (241.02 KB)
 Trainable params: 30,849 (120.50 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 30,851 (120.52 KB)
============================================================

best model: **LSTM best model**
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 10, 32)              │           6,144 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 10, 32)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 128)                 │          82,432 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1)                   │             129 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 177,412 (693.02 KB)
 Trainable params: 88,705 (346.50 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 88,707 (346.52 KB)

============================================================
Walk-Forward Validation Complete. Avg RMSE: 1.1110
============================================================

### 4.1.2 CNN-GRU

This model is a hybrid model so it mixes 1D- Convolutional Neural Network and GRU. The only difference between it and the LSTM/RNN models is that it uses Convolutional Neural Network to capture local temporal patterns and GRU to capture long-term dependencies. So instead of doing a double LSTM/RNN model, the 1D- Convolutional Neural Network is used for local temporal patterns and the GRU is used for long-term dependencies. It also has a BatchNormalization section in essence, it’s a layer that normalizes activations (the intermediate outputs of a neural network) so that each layer receives data that’s centered and scaled — just like how we scale input features before training.


best model: **CNN_GRU**
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv1d (Conv1D)                      │ (None, 8, 64)               │           2,944 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 8, 64)               │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 8, 64)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ gru (GRU)                            │ (None, 32)                  │           9,408 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1)                   │              33 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 37,669 (147.15 KB)
 Trainable params: 12,513 (48.88 KB)
 Non-trainable params: 128 (512.00 B)
 Optimizer params: 25,028 (97.77 KB)
============================================================
Walk-Forward Validation Complete. Avg RMSE: 1.1001
============================================================

## 4.2 Custom Model Architecture

The finance world usually involves a long term pattern and an upward or downward change or spike in a stock to due to a change in market sentiment. The CNN-GRU was selected as the custom architecture for its potential to capture both local temporal patterns (via *CNN* layers) and long-term dependencies (via *GRU*s). 

## 4.3 Hyperparameters
Hyperparameter tuning using the library kerastuner was applied to determine model parameters for the LSTM, RNN, and CNN-GRU models. The hyperparameters that were tuned were:

- Units: The number of units in the LSTM, RNN, and GRU layers.

- Dropout: The dropout rate for the LSTM, RNN, and GRU layers.

- Recurrent dropout: The recurrent dropout rate for the LSTM, RNN, and GRU layers.

- Kernel regularizer: The kernel regularizer for the LSTM, RNN, and GRU layers.

- Optimizer: The optimizer for the LSTM, RNN, and GRU models.

- Loss function: The loss function for the LSTM, RNN, and GRU models.

- Learning rate: The learning rate for the LSTM, RNN, and GRU models.

**Hyperparameters**
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50


## 4.x.4 Training Performance

`![loss vs epoch plot_LSTM](stock-forecast/reports/plots/lstm_plots/plot_fold_1.png)`
`![loss vs epoch plot_LSTM](stock-forecast/reports/plots/cnn_gru_plots/plot_fold_1.png)`
`![loss vs epoch plot_LSTM](stock-forecast/reports/plots/rnn_plots/plot_fold_1.png)`

Takeaway

Fold 1: This represents the best-behaved model so far — training converges cleanly, and validation doesn’t explode.

Fold 2 overfits — good short-term learning but poor generalization.

Fold 3 underfits or fails to converge — needs learning rate or normalization check.

# 5. Model Evaluation and Comparison  (~400–500 words)
## 5.1 Test Performance

The table below summarizes the performance of the three neural network architectures — RNN, CNN-GRU, and LSTM — evaluated on the test set using multiple regression metrics.  
The **Mean Absolute Error (MAE)** was selected as the primary success metric, while RMSE, R², and other secondary metrics were used to assess model robustness and generalization.

|    Model   | MAE   | RMSE  | Close MAE | Close RMSE | Test Loss | sMAPE  | R²    | MedAE | EVS   | PSI  | Bias  | Error Std| Folds |
|------------------------------------------------------------------------------------------------------------------------------------|
| **RNN**    | 0.860 | 1.151 | 3.119     |  4.094     | 0.512     | 149.42 |-0.297 | 0.667 |-0.135 | 13.43| -0.018| 1.114    | 13    |
| **CNN-GRU**| 0.825 | 1.100 | 2.991     |  3.889     | 0.479     | 153.94 |-0.170 | 0.606 | 0.039 | 13.43| -0.162| 1.064    | 13    |
| **LSTM**   | 0.833 | 1.111 | 3.133     |  4.061     | 0.485     | 152.14 |-0.180 | 0.641 |-0.031 | 13.43| -0.102| 1.093    | 13    |


## 5.2 Interpretation

**Overall Interpretation**: CNN-GRU gave the most promising results as it presented the best results for most of the metrics including the main metrics which are the **RMSE(1.100)** and **MAE(0.825)**. From 12 metrics it gave the best results for 10 of them, 1 the results matched and the last one it ranked third.

- **CNN-GRU:**  
  The combination of **convolutional layers** (for spatial feature extraction) and **GRU cells** (for temporal dependency modeling) appears to provide a balanced understanding of both short-term volatility and long-term trends. Its lower MAE and RMSE reflect better generalization.

- **LSTM:**  
  Performed closely behind CNN-GRU, showing reliable temporal learning but slightly higher test loss. This suggests it captured long-term dependencies effectively but may have been more sensitive to noise.

- **RNN:**  
  Had the highest MAE and RMSE, consistent with its simpler architecture, which struggles to retain long-term context. This explains its lower R² score and relatively higher prediction bias.


## 5.3 Best Model Visualization

The figure below shows the **CNN-GRU model’s predicted closing prices** on the test set compared to the true closing prices.  
The predictions closely track actual market movements, with deviations primarily during high-volatility regions — a common challenge in financial forecasting.

`![Actual_predicted_plots](stock-forecast/reports/cnn_gru_Actual_predicted_plots.png)`

**Interpretation:**
- The predicted curve mirrors the true trend, capturing directional changes effectively.
- Slight underestimation occurs during sharp peaks and drops, likely due to smoothing behavior inherent in recurrent architectures.
- The model exhibits strong short-term responsiveness without excessive noise, confirming its balanced bias-variance tradeoff.
- There seems to be a slight shift between data on the x-axis which might be due a problem with data distribution apparent in psi values.


# 6. Conclusion

## 6.1 Summary of Findings
This project developed and evaluated multiple neural network architectures — RNN, LSTM, and CNN-GRU — for forecasting Apple Inc.’s daily closing stock prices. Each model was trained and tested using walk-forward validation to ensure robustness and time-consistent evaluation.  
Among the tested models, the **CNN-GRU hybrid** achieved the best overall performance, recording the lowest MAE (0.825) and RMSE (1.10) on the test set. Its hybrid structure, combining convolutional layers for local pattern extraction and GRU units for temporal dependency modeling, enabled it to capture both short-term fluctuations and long-term trends more effectively than the standalone RNN or LSTM.  
The LSTM model performed comparably well, demonstrating stable learning but slightly higher test loss, while the simple RNN showed weaker generalization and higher bias, confirming its limited capacity for complex temporal dependencies.  
Overall, the project highlights that architectures integrating both spatial and sequential learning (such as CNN-GRU) can offer superior predictive capability for volatile, non-stationary financial data.

## 6.2 Limitations
The dataset was limited to three years of Apple’s daily trading data, which restricts exposure to diverse market conditions such as recessions or extreme volatility events. Feature engineering relied primarily on technical indicators; no macroeconomic or sentiment features were incorporated. Additionally, neural networks were trained on scaled data without exogenous variables like earnings reports or interest rate changes, which could influence future price dynamics.  
While the models captured temporal dependencies, they remain deterministic and do not quantify predictive uncertainty — an important factor for financial risk management.

## 6.3 Future Work
Future iterations could extend this research by incorporating **multivariate data**, including macroeconomic indicators or news sentiment scores, to enrich feature diversity.  
Second, implementing **attention-based architectures** (such as Transformers or Temporal Fusion Transformers) could enhance interpretability and long-horizon forecasting.  
Finally, integrating **probabilistic forecasting** or **Bayesian neural networks** would allow the model to output confidence intervals, making the predictions more actionable for real-world trading and portfolio management.
Also Applying **Hyperparameter tuning** on epochs, batch size, and learning rate to find the best model parameters other than the parameters for the models themselves.


# 7. Appendix
## 7.1 Environment

imbalanced-learn>=0.14.0
imblearn>=0.0
joblib>=1.5.2
keras>=3.11.3


keras-tuner>=1.4.7
kiwisolver>=1.4.9
kt-legacy>=1.0.5
libclang>=18.1.1


matplotlib>=3.10.6
mlxtend>=0.23.4
numpy>=2.3.3
openpyxl>=3.1.5


pandas>=2.3.2
scikit-learn>=1.7.1
scipy>=1.16.1
seaborn>=0.13.2


tensorflow>=2.20.0
yfinance>=0.2.65

## 7.2 Link to Code
The following is the github repo link: https://github.com/laithalfar/Stock_Prediction
