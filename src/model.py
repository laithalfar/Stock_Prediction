import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout
from config import MODEL_DIR
from keras.optimizers import Adam, RMSprop
import kerastuner as kt
from keras.regularizers import l2

# Units is the capacity of the layer (risk of overfitting and slower training when increased)
# Dropout_rate randomly drops connections during training to prevent overfitting.
# Optimizer is the algorithm to adjust weights during training.
# Loss function measures how well the model is performing (mean_squared_error is common for regression tasks).
def create_lstm_model(hp, input_shape):
    """Create an LSTM model for stock prediction."""
    model = Sequential() #Sequential is a linear stack of layers where each layer has exactly one input tensor and one output tensor. That means data flows strictly from the first layer to the last, without branches, skips, or multiple inputs/outputs.
    model.add(LSTM(
    units=hp.Int("units", 32, 128, step=32),
    return_sequences=True,
    input_shape=input_shape,
    recurrent_dropout=hp.Float("rec_dropout", 0.0, 0.3, step=0.1),
    kernel_regularizer=l2(hp.Choice("l2", [0.0, 1e-5, 1e-4]))
))
    model.add(Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
    model.add(LSTM(
    units=hp.Int("units", 32, 128, step=32),
    return_sequences=False,
    input_shape=input_shape,
    recurrent_dropout=hp.Float("rec_dropout", 0.0, 0.3, step=0.1),
    kernel_regularizer=l2(hp.Choice("l2", [0.0, 1e-5, 1e-4]))
))
    model.add(Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
    model.add(Dense(units=1))  # Regression output

    # Hyperparameter tuning for optimizer and learning rate
    optimizer_choice = hp.Choice("optimizer", ["adam", "rmsprop"])
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



def create_recurrent_neural_network(input_shape, units=50, dropout_rate=0.2, optimizer="adam", loss="mean_squared_error"):
    """Create a vanilla RNN (using SimpleRNN)."""
    model = Sequential()
    model.add(SimpleRNN(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss=loss)
    return model


def create_gru_model(input_shape, units=50, dropout_rate=0.2, optimizer="adam", loss="mean_squared_error"):
    """Candidate’s Choice: GRU model (simpler & often faster than LSTM)."""
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss=loss)
    return model
