import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Conv1D, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2

# Units is the capacity of the layer (risk of overfitting and slower training when increased)
# Dropout randomly drops connections during training to prevent overfitting.
# Optimizer is the algorithm to adjust weights during training.
# Loss function measures how well the model is performing (mean_squared_error is common for regression tasks).
def create_lstm_model(hp, input_shape):
    
    """
    Create a LSTM model for stock prediction.

    Parameters:
    - hp: HyperParameters object from hyperopt
    - input_shape: tuple of shape of input data (timesteps, features)

    Returns:
    - model: the created model
    """
    
    model = Sequential() #Sequential is a linear stack of layers where each layer has exactly one input tensor and one output tensor. That means data flows strictly from the first layer to the last, without branches, skips, or multiple inputs/outputs.
    model.add(LSTM(
    units=hp.Int("units_lstm1", 32, 128, step=32), 
    return_sequences=True,
    input_shape=input_shape,
    recurrent_dropout=hp.Float("rec_dropout", 0.0, 0.3, step=0.1),
    kernel_regularizer=l2(hp.Choice("l2", [0.0, 1e-5, 1e-4]))
))
    model.add(Dropout(hp.Float("dropout_lstm1", 0.1, 0.5, step=0.1)))
    model.add(LSTM(
    units=hp.Int("units_lstm2", 32, 128, step=32),
    return_sequences=False,
    input_shape=input_shape,
    recurrent_dropout=hp.Float("rec_dropout", 0.0, 0.3, step=0.1),
    kernel_regularizer=l2(hp.Choice("l2", [0.0, 1e-5, 1e-4]))
))
    model.add(Dropout(hp.Float("dropout_lstm2", 0.1, 0.5, step=0.1)))
    model.add(Dense(units=1))  # Regression output

    # Hyperparameter tuning for optimizer and learning rate
    optimizer_choice = hp.Choice("optimizer_lstm", ["adam", "rmsprop"])
    lr = hp.Choice("lr", [1e-2, 1e-3, 1e-4])

    if optimizer_choice == "adam":
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=hp.Choice("loss", ["mse", "mae", "huber"])
    )

    return model



def create_rnn_model(hp, input_shape):
    
    """
    Create a RNN model for stock prediction.

    Parameters:
    - hp: HyperParameters object from hyperopt
    - input_shape: tuple of shape of input data (timesteps, features)

    Returns:
    - model: the created model
    """

    model = Sequential()
    model.add(SimpleRNN(
        units=hp.Int("units_rnn1", 32, 128, step=32),
        return_sequences=True,
        input_shape=input_shape,
        recurrent_dropout=hp.Float("rec_dropout_rnn1", 0.0, 0.3, step=0.1),
        kernel_regularizer=l2(hp.Choice("l2_rnn1", [0.0, 1e-5, 1e-4]))
    ))

    model.add(Dropout(hp.Float("dropout_rnn1", 0.1, 0.5, step=0.1)))

    # Second RNN layer
    model.add(SimpleRNN(
        units=hp.Int("units_rnn2", 32, 128, step=32),
        return_sequences=False,
        recurrent_dropout=hp.Float("rec_dropout_rnn2", 0.0, 0.3, step=0.1),
        kernel_regularizer=l2(hp.Choice("l2_rnn2", [0.0, 1e-5, 1e-4]))
    ))

    model.add(Dropout(hp.Float("dropout_rnn2", 0.1, 0.5, step=0.1)))

    # Output
    model.add(Dense(units=1))

    # Optimizer
    optimizer_choice = hp.Choice("optimizer_rnn", ["adam", "rmsprop"])
    lr = hp.Choice("lr_rnn", [1e-2, 1e-3, 1e-4])
    optimizer = Adam(learning_rate=lr) if optimizer_choice == "adam" else RMSprop(learning_rate=lr)

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=hp.Choice("loss_rnn", ["mse", "mae", "huber"]),
        metrics=["mae", "mape"]
    )

    return model


def create_cnn_gru_model(hp, input_shape):
    
    """
    Create a CNN-GRU model with hyperparameter tuning.

    Parameters:
    - hp: HyperParameters object from hyperopt
    - input_shape: tuple of shape of input data (timesteps, features)

    Returns:
    - model: the created model
    """
    model = Sequential()
    
    # --- CNN Block ---
    model.add(Conv1D(
        filters=hp.Int("filters", 32, 128, step=32),
        kernel_size=hp.Choice("kernel_size", [2, 3, 5]),
        activation="relu",
        input_shape=input_shape,
        kernel_regularizer=l2(hp.Choice("cnn_l2", [0.0, 1e-5, 1e-4]))
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float("cnn_dropout", 0.1, 0.5, step=0.1)))
    
    # --- GRU Block ---
    model.add(GRU(
        units=hp.Int("gru_units", 32, 128, step=32),
        return_sequences=False,
        recurrent_dropout=hp.Float("gru_rec_dropout", 0.0, 0.3, step=0.1),
        kernel_regularizer=l2(hp.Choice("gru_l2", [0.0, 1e-5, 1e-4]))
    ))
    model.add(Dropout(hp.Float("gru_dropout", 0.1, 0.5, step=0.1)))
    
    # --- Output ---
    model.add(Dense(1, activation="linear"))  # regression output for forecasting
    
    # --- 4. Optimizer tuning ---
    opt_choice = hp.Choice("optimizer_cnn_gru", ["adam", "rmsprop"])
    lr = hp.Choice("lr", [1e-2, 1e-3, 1e-4])
    optimizer = Adam(learning_rate=lr) if opt_choice == "adam" else RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=hp.Choice("loss", ["mse", "mae", "huber"])
    )

    
    return model