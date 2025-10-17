import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (LSTM, SimpleRNN, GRU, Dense, Dropout, Conv1D, 
                          BatchNormalization, Attention, Add, Input, 
                          Bidirectional, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D)
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2

# Units is the capacity of the layer (risk of overfitting and slower training when increased)
# Recurrent dropout: randomly drops connections within the temporal loop of the RNN (between timesteps). This prevents the model from 
# memorizing specific sequences and forces it to learn more general temporal dependencies. Recurrent dropout adds controlled forgetting, improving generalization and robustness to noise.
# A kernel regularizer adds a small penalty to the network’s weights during training to discourage them from growing too large. Large weights usually indicate overfitting — the model’s trying too hard to fit exact 
# training examples instead of learning general relationships.

# Dropout randomly drops connections during training to prevent overfitting.
# Optimizer is the algorithm to adjust weights during training.
# Loss function measures how well the model is performing (mean_squared_error is common for regression tasks).
def create_lstm_model(hp, input_shape):
    
    """
    Create a LSTM model for stock prediction.

    Parameters:
    - hp: HyperParameters object from keras_tuner
    - input_shape: tuple of shape of input data (timesteps, features)

    Returns:
    - model: the created model
    """
    
    # Hyperparameters
    n_layers = hp.Int("n_layers_lstm", 2, 4, step=1) # building 2-4 layers
    use_attention = hp.Boolean("use_attention_lstm", default = False) # give attention option
    use_bidirectional = hp.Boolean("use_bidirectional_lstm", default = False) # give bidirectional option
    
    model = Sequential()

    # First LSTM layer
    if use_bidirectional:
        model.add(Bidirectional( #wrap LSTM with bidirectionality
            LSTM(
                units=hp.Int("units_lstm1", 32, 256, step=16),
                return_sequences=True,
                input_shape=input_shape,
                recurrent_dropout=hp.Float("rec_dropout_lstm1", 0.0, 0.3, step=0.1),
                kernel_regularizer=l2(hp.Choice("l2_lstm1", [0.0, 1e-5, 1e-4, 1e-3]))
            ), 
            #input_shape=input_shape
        ))
    else:
        model.add(LSTM( # else keep a standard LSTM layer
            units=hp.Int("units_lstm1", 32, 256, step=16),
            return_sequences=True,
            input_shape=input_shape,
            recurrent_dropout=hp.Float("rec_dropout_lstm1", 0.0, 0.3, step=0.1),
            kernel_regularizer=l2(hp.Choice("l2_lstm1", [0.0, 1e-5, 1e-4, 1e-3]))
        ))

    model.add(LayerNormalization()) #is a normalization layer that standardizes each sample independently by using the statistics of its own features.It plays nicely with RNNs/LSTMs/Transformers because it normalizes within each time step’s feature vector.
    model.add(Dropout(hp.Float("dropout_lstm1", 0.1, 0.5, step=0.1)))
    
    # Additional LSTM layers (variable depth)
    for i in range(2, n_layers + 1): # loop through the layers
        return_seq = True if i < n_layers else False # if it's not the last layer, return sequences, else return the final output
        model.add(LSTM(
            units=hp.Int(f"units_lstm{i}", 32, 256, step=16),
            return_sequences=return_seq,
            recurrent_dropout=hp.Float(f"rec_dropout_lstm{i}", 0.0, 0.3, step=0.1),
            kernel_regularizer=l2(hp.Choice(f"l2_lstm{i}", [0.0, 1e-5, 1e-4, 1e-3]))
        ))
        model.add(LayerNormalization())
        model.add(Dropout(hp.Float(f"dropout_lstm{i}", 0.1, 0.5, step=0.1)))
    
    # Attention mechanism (if enabled)
    # Note: True attention requires Functional API, so we use a simple dense layer as proxy
    if use_attention:
        model.add(Dense(hp.Int("attention_units", 32, 128, step=16), activation='tanh'))
    
    # Dense output
    model.add(Dense(units=1))

    # Hyperparameter tuning for optimizer and learning rate
    optimizer_choice = hp.Choice("optimizer_lstm", ["adam", "rmsprop"])
    lr = hp.Choice("lr_lstm", [1e-2, 1e-3, 1e-4])

    if optimizer_choice == "adam":
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=hp.Choice("loss_lstm", ["mse", "mae", "huber"]),
        metrics=["mae", "mape"]
    )

    return model



def create_rnn_model(hp, input_shape):
    """
    Enhanced RNN model with variable depth and bidirectional option.

    Parameters:
    - hp: HyperParameters object from keras_tuner
    - input_shape: tuple of shape of input data (timesteps, features)

    Returns:
    - model: compiled Keras model
    """
    
    # Hyperparameters
    n_layers = hp.Int("n_layers_rnn", 2, 4, step=1)
    use_attention = hp.Boolean("use_attention_rnn", default = False) # give attention option
    use_bidirectional = hp.Boolean("use_bidirectional_rnn", default = False)
    
    model = Sequential()
    
    # First RNN layer
    if use_bidirectional:
        model.add(Bidirectional(
            SimpleRNN(
                units=hp.Int("units_rnn1", 32, 256, step=16),
                return_sequences=True,
                input_shape=input_shape,
                recurrent_dropout=hp.Float("rec_dropout_rnn1", 0.0, 0.3, step=0.1),
                kernel_regularizer=l2(hp.Choice("l2_rnn1", [0.0, 1e-5, 1e-4, 1e-3]))
            ),
            #input_shape=input_shape
        ))
    else:
        model.add(SimpleRNN(
            units=hp.Int("units_rnn1", 32, 256, step=16),
            return_sequences=True,
            input_shape=input_shape,
            recurrent_dropout=hp.Float("rec_dropout_rnn1", 0.0, 0.3, step=0.1),
            kernel_regularizer=l2(hp.Choice("l2_rnn1", [0.0, 1e-5, 1e-4, 1e-3]))
        ))
    
    model.add(LayerNormalization())
    model.add(Dropout(hp.Float("dropout_rnn1", 0.1, 0.5, step=0.1)))
    
    # Additional RNN layers (variable depth)
    for i in range(2, n_layers + 1):
        return_seq = True if i < n_layers else False
        model.add(SimpleRNN(
            units=hp.Int(f"units_rnn{i}", 32, 256, step=16),
            return_sequences=return_seq,
            recurrent_dropout=hp.Float(f"rec_dropout_rnn{i}", 0.0, 0.3, step=0.1),
            kernel_regularizer=l2(hp.Choice(f"l2_rnn{i}", [0.0, 1e-5, 1e-4, 1e-3]))
        ))
        model.add(LayerNormalization())
        model.add(Dropout(hp.Float(f"dropout_rnn{i}", 0.1, 0.5, step=0.1)))

    # Attention mechanism (if enabled)
    # Note: True attention requires Functional API, so we use a simple dense layer as proxy
    if use_attention:
        model.add(Dense(hp.Int("attention_units_rnn", 32, 128, step=16), activation='tanh'))

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
    Advanced CNN-GRU hybrid with residual connections and attention.
    Uses Functional API for skip connections.

    Parameters:
    - hp: HyperParameters object from keras_tuner
    - input_shape: tuple of shape of input data (timesteps, features)

    Returns:
    - model: compiled Keras model
    """
    
    # Hyperparameters
    n_cnn_layers = hp.Int("n_cnn_layers", 1, 3, step=1)
    n_gru_layers = hp.Int("n_gru_layers", 1, 3, step=1)
    use_ln_in_cnn = hp.Boolean("use_ln_in_cnn", default=False)
    use_residual = hp.Boolean("use_residual_cnn_gru", default = True)
    use_attention = hp.Boolean("use_attention_cnn_gru", default = False)
    
    # Input layer
    inputs = Input(shape=input_shape)
    x = inputs
    
    # --- CNN Block for local pattern extraction ---
    for i in range(n_cnn_layers):
        filters = hp.Int(f"filters_cnn{i+1}", 32, 256, step=16)
        kernel_size = hp.Choice(f"kernel_size_cnn{i+1}", [2, 3, 5])
        
        cnn_out = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",  # For residual connections
            kernel_regularizer=l2(hp.Choice(f"cnn_l2_{i+1}", [0.0, 1e-5, 1e-4, 1e-3]))
        )(x)
        cnn_out = LayerNormalization()(cnn_out) if use_ln_in_cnn else BatchNormalization()(cnn_out)
        cnn_out = Dropout(hp.Float(f"cnn_dropout_{i+1}", 0.1, 0.5, step=0.1))(cnn_out)
        
        # Residual connection (if enabled and shapes match)
        if use_residual:
            if x.shape[-1] == cnn_out.shape[-1]:
                x = Add()([x, cnn_out])
            else:
                # Project x to match cnn_out dimensions
                proj = Conv1D(filters=cnn_out.shape[-1], kernel_size=1, padding="same")(x)
                x = Add()([proj, cnn_out])
        else: 
            x = cnn_out
    
    # --- GRU Block for temporal dependencies ---
    for i in range(n_gru_layers):
        return_seq = True if (i < n_gru_layers - 1 or use_attention) else False
        
        gru_out = GRU(
            units=hp.Int(f"gru_units_{i+1}", 32, 256, step=16),
            return_sequences=return_seq,
            recurrent_dropout=hp.Float(f"gru_rec_dropout_{i+1}", 0.0, 0.3, step=0.1),
            kernel_regularizer=l2(hp.Choice(f"gru_l2_{i+1}", [0.0, 1e-5, 1e-4, 1e-3]))
        )(x)
        x = LayerNormalization()(gru_out)
        x = Dropout(hp.Float(f"gru_dropout_{i+1}", 0.1, 0.5, step=0.1))(x)
    
    # --- Attention Mechanism (if enabled) ---
    if use_attention and len(x.shape) == 3:  # Check if sequences remain
        # Self-attention
        attention_output = MultiHeadAttention(
            num_heads=hp.Int("attention_heads", 2, 8, step=2),
            key_dim=hp.Int("attention_key_dim", 16, 64, step=16)
        )(x, x)
        x = Add()([x, attention_output])  # Residual around attention
        x = LayerNormalization()(x)
        # Global average pooling to flatten
        x = GlobalAveragePooling1D()(x)
    
    # --- Dense layers ---
    x = Dense(
        units=hp.Int("dense_units", 32, 128, step=16),
        activation="relu",
        kernel_regularizer=l2(hp.Choice("dense_l2", [0.0, 1e-5, 1e-4]))
    )(x)
    x = Dropout(hp.Float("dense_dropout", 0.1, 0.4, step=0.1))(x)
    
    # Output
    outputs = Dense(1, activation="linear")(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Optimizer
    opt_choice = hp.Choice("optimizer_cnn_gru", ["adam", "rmsprop"])
    lr = hp.Choice("lr_cnn_gru", [1e-2, 1e-3, 1e-4, 5e-5])
    optimizer = Adam(learning_rate=lr) if opt_choice == "adam" else RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=hp.Choice("loss_cnn_gru", ["mse", "mae", "huber"]),
        metrics=["mae", "mape"]
    )

    return model