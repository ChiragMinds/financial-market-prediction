# src/model.py
"""
Model module: builds the Keras model architecture used in the notebook.
Architecture: Conv1D -> MaxPool -> Bidirectional LSTMs -> Attention -> Dense
Includes the custom directional-aware loss used in the notebook.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, Dropout, Bidirectional, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


def profit_directional_loss(y_true, y_pred):
    """
    Custom loss: MSE + direction mismatch penalty.
    y_true and y_pred shape: (batch, steps_ahead)
    """
    mse = K.mean(K.square(y_true - y_pred))
    # Directional difference penalty (compare signs of successive differences)
    # for steps_ahead > 1
    direction_penalty = 0.0
    if K.ndim(y_true) == 2 and K.shape(y_true)[1] > 1:
        true_diff = y_true[:, 1:] - y_true[:, :-1]
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        # mismatch count where sign differs
        sign_mismatch = K.not_equal(K.sign(true_diff), K.sign(pred_diff))
        # cast to float and average
        direction_penalty = K.mean(K.cast(sign_mismatch, K.floatx()))
    return mse + 0.5 * direction_penalty


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W = Dense(1, activation='tanh')

    def call(self, inputs):
        # inputs shape: (batch, time, features)
        scores = self.W(inputs)                       # (batch, time, 1)
        weights = tf.nn.softmax(scores, axis=1)       # (batch, time, 1)
        context = tf.reduce_sum(inputs * weights, axis=1)  # (batch, features)
        return context


def build_model(window_size=50, steps_ahead=5, lr=1e-3):
    """
    Build and compile the model.

    Returns:
      model: a compiled tf.keras.Model
    """
    input_layer = Input(shape=(window_size, 1))

    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)

    x = Attention()(x)
    x = Dropout(0.3)(x)

    output = Dense(steps_ahead)(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss=profit_directional_loss)

    return model
