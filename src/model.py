# src/model.py
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, Dropout, Bidirectional, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def profit_directional_loss(y_true, y_pred):
    """
    Custom loss used in notebook:
    MSE + 0.5 * direction mismatch penalty
    """
    mse = K.mean(K.square(y_true - y_pred))
    # compute directional differences along steps dimension
    true_diff = y_true[:, 1:] - y_true[:, :-1]
    pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    mismatch = K.cast(K.not_equal(K.sign(true_diff), K.sign(pred_diff)), K.floatx())
    direction_loss = K.mean(mismatch)
    return mse + 0.5 * direction_loss

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W = Dense(1, activation='tanh')

    def call(self, inputs):
        # inputs: (batch, time, features)
        scores = self.W(inputs)                       # (batch, time, 1)
        weights = tf.nn.softmax(scores, axis=1)       # (batch, time, 1)
        context = tf.reduce_sum(inputs * weights, axis=1)  # (batch, features)
        return context

def build_model(window_size=50, steps_ahead=5, lr=1e-3):
    """
    Build the model used in the notebook:
      Conv1D -> MaxPool -> Bi-LSTM stack -> Attention -> Dense output
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
