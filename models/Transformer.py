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

        return self.feedforward(out)


class FeedForward(layers.Layer):
    def __init__(self, ff_dim, target_dim, rate=0):
        super(FeedForward, self).__init__()

        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation='relu'),
                layers.Dense(target_dim, activation='relu'),
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
        N_LAYERS = 2
        N_HEADS = 8
        FF_DIM = 128
        N_BLOCKS = 2
        EMBED_DIM = 32
        DROPUT_RATE = 0.05
        TIME_2_VEC_DIM = 5
        SKIP_CONNECTION_STRENGTH = 0.2

        inp = Input((N_INPUTS, N_FEATURES), name='previous info')
        inp2 = Input((N_OUTPUTS, N_PREDICT_INFO), name='predict info')
        x = inp
        y = inp2

        # time_embedding using time2vec
        time_embedding = layers.TimeDistributed(time2vec(TIME_2_VEC_DIM - 1))(x[:, :, -1:])
        x = Concatenate(axis=-1)([x, time_embedding])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        N_PREDICT_FEAT = y.shape[-1]
        # transformer encoder part
        for k in range(N_BLOCKS * 2):
            x_old = x
            transformer_block = TransformerBlock(EMBED_DIM, N_FEATURES + TIME_2_VEC_DIM, N_HEADS,
                                                 FF_DIM, DROPUT_RATE)
            x = transformer_block(x)
            x = ((1.0 - SKIP_CONNECTION_STRENGTH) * x) + (SKIP_CONNECTION_STRENGTH * x_old)

        # transformer decoder part
        ### decoder self attention
        # decoder_mask = create_look_ahead_mask(N_OUTPUTS)
        for layer in range(N_LAYERS):
            for k in range(N_BLOCKS):
                y_old = y
                transformer_block = TransformerBlock(EMBED_DIM, N_PREDICT_FEAT, N_HEADS, FF_DIM,
                                                     DROPUT_RATE)
                y = transformer_block(inputs=y)
                y = ((1.0 - SKIP_CONNECTION_STRENGTH) * y) + (SKIP_CONNECTION_STRENGTH * y_old)
            ### decoder encoder-decoder-attention
            y_old = y
            y = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=EMBED_DIM) \
                (query=y, value=x, key=x)
            y = layers.LayerNormalization(epsilon=1e-6)(y_old + y)

            feedfoward = FeedForward(ff_dim=FF_DIM, target_dim=N_PREDICT_FEAT)
            y = feedfoward(y)

        # y = layers.Dense(FF_DIM, activation='relu')(y)
        output = layers.Dense(1)(y)
        output = layers.Flatten()(output)
        # output = tf.transpose(y, [0, 2, 1])
        # output = layers.LSTM(N_OUTPUTS)(output)
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
        self.model.compile(loss=keras.losses.Huber(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate,
                                                              beta_1=0.9,
                                                              beta_2=0.98,
                                                              epsilon=1e-9),
                           metrics=[new_mape])
    @staticmethod
    def predict(model, enc_inp, dec_inp):

        N_OUPUTS = dec_inp.shape[1]
        for i in range(1, N_OUPUTS):
            predictions = model([enc_inp, dec_inp], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, i]  # (batch_size, 1, vocab_size)

            dec_inp = dec_inp.numpy()
            dec_inp[:, i, -1] = predictions
            dec_inp = tf.convert_to_tensor(dec_inp)

        output = dec_inp[:, :, -1].numpy()
        return output


if __name__ == '__main__':
    mask = create_look_ahead_mask(180)
    print(mask)
    # model = get_transformer_model(54)
    # print(model.summary())
