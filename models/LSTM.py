from config import *
from models.Time2Vec import time2vec
from models.Losses import *


class LSTMTime2Vec():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = None

    def nn_sctructure(self, N_INPUTS, N_FEATURES, N_OUTPUTS):
        N_BLOCKS = 128
        inp = Input((N_INPUTS, N_FEATURES))
        x = inp

        time_embedding = TimeDistributed(time2vec(3))(x[:, :, -1:])
        x = Concatenate(axis=-1)([x, time_embedding])

        x = LSTM(N_BLOCKS)(x)

        x = RepeatVector(N_OUTPUTS)(x)

        x = LSTM(N_BLOCKS,  return_sequences=True)(x)

        x = LSTM(N_BLOCKS)(x)

        x = Dense(N_OUTPUTS, activation='gelu')(x)

        out = x
        return inp, out

    def build_model(self):
        N_INPUTS = self.x.shape[1]
        N_FEATURES = self.x.shape[2]
        N_OUTPUTS = self.y.shape[1]
        inputs, outputs = self.nn_sctructure(N_INPUTS, N_FEATURES, N_OUTPUTS)
        self.model = Model(inputs, outputs)
        self.model.compile(loss='mse', optimizer='adam', metrics=[new_mape])


class LSTMTime2VecDeeper():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = None

    def nn_sctructure(self, N_INPUTS, N_FEATURES, N_OUTPUTS):
        N_BLOCKS = 128
        inp = Input((N_INPUTS, N_FEATURES))
        x = inp

        time_embedding = TimeDistributed(time2vec(5))(x[:, :, -1:])
        x = Concatenate(axis=-1)([x, time_embedding])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        x = LSTM(N_BLOCKS, return_sequences=True)(x)

        x = LSTM(N_BLOCKS, return_sequences=True)(x) + x
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        x = LSTM(N_BLOCKS)(x)

        x = RepeatVector(N_OUTPUTS)(x)

        x = LSTM(N_BLOCKS, return_sequences=True)(x)

        x = LSTM(N_BLOCKS, return_sequences=True)(x) + x
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        x = LSTM(N_BLOCKS)(x)

        x = Dense(N_OUTPUTS, activation='selu')(x)

        out = x
        return inp, out

    def build_model(self):
        N_INPUTS = self.x.shape[1]
        N_FEATURES = self.x.shape[2]
        N_OUTPUTS = self.y.shape[1]
        inputs, outputs = self.nn_sctructure(N_INPUTS, N_FEATURES, N_OUTPUTS)
        self.model = Model(inputs, outputs)
        self.model.compile(loss='mae', optimizer='adam', metrics=[new_mape])

class ResidualWrapper(Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, training=None, mask=None):
        delta = self.model(inputs)
        return inputs + delta