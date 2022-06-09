from config import *
from models.Losses import *


class ConvolutionNetwork():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def nn_structure(self):
        CONV_WIDTH = 5
        inp_shape1 = self.x.shape[1]
        inp_shape2 = self.x.shape[2]
        OUT_STEPS = self.y.shape[1]

        inp = Input((inp_shape1, inp_shape2))
        x = inp
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        # x = tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :])(x)
        # Shape => [batch, 1, conv_units]
        x = tf.keras.layers.Conv1D(64, activation='relu', kernel_size=(CONV_WIDTH))(x)
        # x = tf.keras.layers.Conv1D(128, activation='relu', kernel_size=(CONV_WIDTH))(x)
        # x = tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH))(x)
        # tf.keras.layers.Conv1D(512, activation='relu', kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        x = tf.keras.layers.Dense(OUT_STEPS,
                              kernel_initializer=tf.initializers.zeros())(x)
        x = tf.keras.layers.Flatten()(x)

        out = x

        return inp, out

    def build_model(self):
        inputs, outputs = self.nn_structure()
        self.model = Model(inputs, outputs)
        self.model.compile(loss='mse', optimizer='adam', metrics=[new_mape])
