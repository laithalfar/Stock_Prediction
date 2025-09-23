from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout

def create_lstm_model(input_shape, units=50, dropout_rate=0.2, optimizer="adam", loss="mean_squared_error"):
    """Create an LSTM model for stock prediction."""
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))  # Regression output
    model.compile(optimizer=optimizer, loss=loss)
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
