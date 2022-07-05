from config import *
from models.Time2Vec import time2vec
from models.Losses import *
from models.LearningRateWarmUp import CustomSchedule


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class TransformerBlock(layers.Layer):
    name = ''

    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate)
        self.feedforward = FeedForward(ff_dim, feat_dim)

    def call(self, inputs, mask=None, training=True):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout(attn_output, training=training)
        out = self.layernorm(inputs + attn_output)
        out = self.feedforward(out)
        out = tf.multiply((1.0 - 0.5), out) + \
            tf.multiply(0.5, inputs)
        return out


class FeedForward(layers.Layer):
    def __init__(self, ff_dim, target_dim, rate=0):
        super(FeedForward, self).__init__()

        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation='gelu'),
                layers.Dense(target_dim, activation='gelu'),
            ]
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate)

    def call(self, inputs, training, **kwargs):
        ffn_output = self.ffn(inputs)
        ffn_output = self.dropout(ffn_output)

        return self.layernorm(inputs + ffn_output)


class Transformer():

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.model = None

    def nn_sctructure(self, N_INPUTS, N_FEATURES, N_OUTPUTS, N_PREDICT_INFO):
        # TRANSFORMER + TIME2VEC
        N_LAYERS = 1
        N_HEADS = 8
        FF_DIM = 128
        N_BLOCKS = 2
        EMBED_DIM = 32
        DROPUT_RATE = 0.05

        inp = Input((N_INPUTS, N_FEATURES), name='previous info')
        inp2 = Input((N_OUTPUTS, N_PREDICT_INFO), name='predict info')
        print(inp)
        x = inp
        y = inp2

        # time_embedding using time2vec
        # time_embedding = layers.TimeDistributed(time2vec(TIME_2_VEC_DIM - 1))(x[:, :, -1:])
        # x = Concatenate(axis=-1)([x, time_embedding])
        # x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        N_PREDICT_FEAT = y.shape[-1]
        # transformer encoder part
        for k in range(N_BLOCKS):
            transformer_block = TransformerBlock(EMBED_DIM, (N_FEATURES), N_HEADS,
                                                 FF_DIM, DROPUT_RATE)
            transformer_block.name = 'ENC_BLOCK' + str(k)
            x = transformer_block(x)


        # transformer decoder part
        ### decoder self attention
        # decoder_mask = create_look_ahead_mask(N_OUTPUTS)
        for layer in range(N_LAYERS):
            for k in range(N_BLOCKS):
                transformer_block = TransformerBlock(EMBED_DIM, N_PREDICT_FEAT, N_HEADS, FF_DIM,
                                                     DROPUT_RATE)
                transformer_block.name = 'DEC_BLOCK' + str(k)
                y = transformer_block(inputs=y)

            ### decoder encoder-decoder-attention
            y_old = y
            y = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=EMBED_DIM) \
                (query=y, value=x, key=x)
            y = layers.LayerNormalization(epsilon=1e-6)(y_old + y)

            feedfoward = FeedForward(ff_dim=FF_DIM, target_dim=N_PREDICT_FEAT)
            y = feedfoward(y)

        y = layers.Dense(FF_DIM, activation='selu')(y)
        output = layers.Dense(1)(y)
        output = layers.Flatten()(output)

        out = output

        return inp, inp2, out

    def build_model(self):
        N_INPUTS = self.x.shape[1]
        N_FEATURES = self.x.shape[2]
        N_OUTPUTS = self.y.shape[1]
        N_PREDICT_INFO = self.z.shape[2]
        inputs, inputs2, outputs = self.nn_sctructure(N_INPUTS,
                                                      N_FEATURES,
                                                      N_OUTPUTS,
                                                      N_PREDICT_INFO)
        self.model = Model([inputs, inputs2], outputs)
        learning_rate = CustomSchedule(N_FEATURES)
        # keras.losses.Huber()    tfa.losses.PinballLoss(tau=0.9)
        self.model.compile(loss=keras.losses.Huber(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate,
                                                              beta_1=0.9,
                                                              beta_2=0.98,
                                                              epsilon=1e-9),
                           metrics=[new_mape])


if __name__ == '__main__':
    x = np.zeros((1, 150, 60))
    y = np.zeros((1, 180))
    z = np.zeros((1, 180, 6))
    model = Transformer(x, y, z)
    model.build_model()
    print(model.model.summary())
    # model = get_transformer_model(54)
    # print(model.summary())
