from config import *
from models.Time2Vec import time2vec
from models.Losses import *

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.4):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


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

        x = LSTM(N_BLOCKS, return_sequences=True)(x)

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


class LSTMTime2VecMultiInput():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.model = None

    def nn_sctructure(self, N_INPUTS, N_FEATURES, N_PREDICT_INFO, N_OUTPUTS):
        N_BLOCKS = 128
        inp1 = Input((N_INPUTS, N_FEATURES), name='previous info')
        inp2 = Input((N_OUTPUTS, N_PREDICT_INFO), name='predict info')
        x = inp1
        y = inp2

        time_embedding = TimeDistributed(time2vec(5))(x[:, :, -1:])
        x = Concatenate(axis=-1)([x, time_embedding])

        x = LSTM(N_BLOCKS, return_sequences=True)(x)

        x = LSTM(N_BLOCKS)(x)

        x = RepeatVector(N_OUTPUTS)(x)

        # y = Dense(32, activation='tanh')(y)
        x = Concatenate(axis=-1)([x, y])

        x = LSTM(N_BLOCKS, return_sequences=True)(x)

        x = LSTM(N_BLOCKS, return_sequences=True)(x)

        x = TimeDistributed(Dense(64, activation='relu'))(x)

        x = TimeDistributed(Dense(1, activation='gelu', bias_regularizer='l2'))(x)

        out = x
        return inp1, inp2, out

    def build_model(self):
        N_INPUTS = self.x.shape[1]
        N_FEATURES = self.x.shape[2]
        N_OUTPUTS = self.y.shape[1]
        N_PREDICT_INFO = self.z.shape[-1]
        inputs1, inputs2, outputs = self.nn_sctructure(N_INPUTS, N_FEATURES,
                                                       N_PREDICT_INFO,
                                                       N_OUTPUTS)
        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        self.model.compile(loss=keras.losses.Huber(), optimizer='adam', metrics=[new_mape])
        # self.model.compile(loss='mse', optimizer='adam', metrics=[new_mape])


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

        # x = LSTM(N_BLOCKS)(x)
        #
        # x = Dense(N_OUTPUTS, activation='selu')(x)

        x = TimeDistributed(Dense(1, activation='gelu'))(x)

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
