from config import *
from models.Time2Vec import time2vec
from models.Losses import *


class RNN():

    def __init__(self, input_shape, output_shape, num_hiddens=30, num_layers=4):
        super().__init__()
        # Embedding layer
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.rnn_encoder = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens)
             for _ in range(num_layers)]), return_sequences=True,
            return_state=True)
        self.rnn_decoder = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens)
             for _ in range(num_layers)]), return_sequences=True,
            return_state=True)
        self.dense1 = tf.keras.layers.Dense(output_shape, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_shape, activation='gelu')

        self.model = self.build_model()

    def nn_structure(self):
        inp = Input(self.input_shape)
        x = inp
        enc_output = self.rnn_encoder(x)
        x = enc_output[0]
        state = enc_output[1:]
        context = tf.repeat(tf.expand_dims(state[-1], axis=1), repeats=x.shape[1], axis=1)
        X_and_context = tf.concat((x, context), axis=2)
        rnn_output = self.rnn_decoder(X_and_context, state)

        output = self.dense1(rnn_output[0])
        x = layers.GlobalAveragePooling1D(data_format="channels_last")(output)
        output = self.dense2(x)
        return inp, output

    def build_model(self):
        inp, out = self.nn_structure()
        model = Model(inp, out)
        model.compile(loss='mae', optimizer='adam', metrics=[new_mape])
        return model


if __name__ == '__main__':
    rnn = RNN((30, 10), 180)

    print(rnn.model.summary())
