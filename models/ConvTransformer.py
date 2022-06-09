from config import *

def transform(example):
    return example.numpy()

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(feat_dim, activation="relu")])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        self._trainable = True

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


class SelfAttentionConv(layers.Layer):
    def __init__(self, k, t, headers=8, kernel_size=5, mask_next=True, mask_diag=False):
        super(SelfAttentionConv, self).__init__()
        self._trainable = True
        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag
        self.t = t
        h = headers

        padding = (kernel_size - 1)

        self.toqueries = layers.Conv1D(filters=k * h, padding='same', kernel_size=kernel_size, \
                                       use_bias=True, input_shape=(20,68))
        self.tokeys = layers.Conv1D(filters=k * h, padding='same', kernel_size=kernel_size,
                                    use_bias=True, input_shape=(20,68))
        self.tovalues = layers.Conv1D(filters=k * h, padding='same', kernel_size=1,
                                      use_bias=True, input_shape=(20,68))

        self.unifyheads = layers.Dense(k)

    def call(self, inputs, *args, **kwargs):

        k = self.k
        t = self.t
        # checking Embedding dimension
        assert self.k == k, \
            'Number of time series ' + str(k) + ' didnt match the number of k ' + str(
                self.k) + ' in the initialization of the attention layer.'
        h = self.headers

        # Transpose to see the different time series as different channels
        # inputs = tf.transpose(inputs, [0, 2, 1])
        queries = self.toqueries(inputs)
        queries = tf.reshape(self.toqueries(inputs), [-1, k, h, t])

        keys = tf.reshape(self.tokeys(inputs), [-1, k, h, t])
        values = tf.reshape(self.tovalues(inputs), [-1, k, h, t])

        # Transposition to return the canonical format
        queries = tf.transpose(queries, [0, 2, 3, 1])
        # batch, header, time step, time serie (b, h, t, k)

        values = tf.transpose(values, [0, 2, 3, 1])
        # batch, header, time step, time serie (b, h, t, k)

        keys = tf.transpose(keys, [0, 2, 3, 1])
        # batch, header, time step, time serie (b, h, t, k)

        # Weights
        queries = queries / (k ** (.25))
        keys = keys / (k ** (.25))

        queries = tf.reshape(tf.transpose(queries, [0, 2, 1, 3]), [-1, t, k])
        keys = tf.reshape(tf.transpose(keys, [0, 2, 1, 3]), [-1, t, k])
        values = tf.reshape(tf.transpose(values, [0, 2, 1, 3]), [-1, t, k])

        weights = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))

        # ## Mask the upper & diag of the attention matrix
        # if self.mask_next:
        #     if self.mask_diag:
        #         indices = torch.triu_indices(t, t, offset=0)
        #         weights[:, indices[0], indices[1]] = float('-inf')
        #     else:
        #         indices = torch.triu_indices(t, t, offset=1)
        #         weights[:, indices[0], indices[1]] = float('-inf')

        # Softmax
        weights = layers.Softmax(2)(weights)

        # Output
        output = tf.matmul(weights, values)
        output = tf.reshape(output, [-1, h, t, k])
        output = tf.reshape(tf.transpose(output, [0, 2, 1, 3]), [-1, t, k * h])

        return self.unifyheads(output)  # shape (b,t,k)


class ConvTransformerBlock(layers.Layer):

    def __init__(self, k, t, headers, kernel_size=5, mask_next=True, mask_diag=False,
                 dropout_proba=0.2):
        super(ConvTransformerBlock, self).__init__()

        self.attention = SelfAttentionConv(k, t, headers, kernel_size, mask_next, mask_diag)

        # first & second layer norm
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        # feed forward network
        self.fc1 = layers.Dense(4 * k, activation='relu')
        self.fc2 = layers.Dense(k, activation='relu')

    def call(self, inputs, *args, **kwargs):
        # self attention + residual
        x = self.attention(inputs) + inputs

        # first normalization
        x = self.norm1(x)

        # feed forward network + residual
        oldx = x

        x = self.fc1(x)
        x = self.fc2(x)

        x = x + oldx

        # second Normalization
        x = self.norm2(x)

        return x


class ConvTransformer(keras.Model):

    def __init__(self, headers, seq_length, feature_len, target_seq_length, depth=3,
                 kernel_size=5, mask_next=True, mask_diag=False, dropout_proba=0.2,
                 num_tokens=None):
        super(ConvTransformer, self).__init__()
        # layers.Embedding()
        self.k = feature_len
        self.seq_length = seq_length

        # transformer blocks
        tblocks = []
        for t in range(depth):
            tblocks.append(
                ConvTransformerBlock(feature_len, seq_length, headers, kernel_size,
                                     mask_next, mask_diag, dropout_proba))

        self.TransformerBlocks = keras.Sequential(tblocks)

        # transform seq_length to target seq_length
        self.dense = layers.Dense(target_seq_length, activation='gelu')

    def call(self, inputs, training=None, mask=None):
        x = self.TransformerBlocks(inputs)
        x = self.dense(layers.Flatten()(x))

        return x


if __name__ == '__main__':
    model = ConvTransformer(8, 60, 56, 180)
    model.compile(loss='mse', optimizer='adam')
    sample_data = tf.zeros((128, 60, 56))
    output = model(sample_data)
    print(model.summary())
    print(output.shape)
