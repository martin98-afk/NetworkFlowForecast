from config import *
from models.Time2Vec import time2vec
from models.Losses import *


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(feat_dim), ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


class Transfomer():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = None

    def nn_sctructure(self, N_INPUTS, N_FEATURES, N_OUTPUTS):
        # TRANSFORMER + TIME2VEC
        N_HEADS = 8
        FF_DIM = 128
        N_BLOCKS = 2
        EMBED_DIM = 32
        DROPUT_RATE = 0.05
        TIME_2_VEC_DIM = 5
        SKIP_CONNECTION_STRENGTH = 0.1

        inp = Input((N_INPUTS, N_FEATURES))
        x = inp

        time_embedding = layers.TimeDistributed(time2vec(TIME_2_VEC_DIM - 1))(x[:, :, -1:])
        x = Concatenate(axis=-1)([x, time_embedding])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        for k in range(N_BLOCKS):
            x_old = x
            transformer_block = TransformerBlock(EMBED_DIM, N_FEATURES + TIME_2_VEC_DIM, N_HEADS,
                                                 FF_DIM, DROPUT_RATE)
            x = transformer_block(x)
            x = ((1.0 - SKIP_CONNECTION_STRENGTH) * x) + (SKIP_CONNECTION_STRENGTH * x_old)

        x = layers.LayerNormalization()(layers.MultiHeadAttention(num_heads=N_HEADS,
                                                                  key_dim=EMBED_DIM)(x, x) + x)

        x = keras.Sequential([layers.Dense(FF_DIM, activation="relu"), layers.Dense(
            N_OUTPUTS*2, activation="relu")])(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
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
