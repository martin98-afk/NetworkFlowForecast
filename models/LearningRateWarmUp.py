from config import *

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=100):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    learning_rate = CustomSchedule(d_model=32)
    plt.plot(learning_rate(tf.range(300, dtype=tf.float32)))
    plt.ylabel('Learning Rate')
    plt.xlabel('Train Step')
    plt.show()