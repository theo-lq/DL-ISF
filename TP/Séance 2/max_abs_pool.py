import tensorflow as tf
from tensorflow import keras

class MaxAbsPool2D(keras.layers.Layer):

    def __init__(self, pool_size, padding="same", **kwargs):
        self.pool_size = pool_size if type(pool_size) == tuple else (pool_size, pool_size)
        self.padding = padding
        super(MaxAbsPool2D, self).__init__(**kwargs)

    def compute_output_shape(self, in_shape):
        if self.padding == "same":
            shape = (in_shape[0],
                    tf.math.ceil(in_shape[1] / self.pool_size[0]),
                    tf.math.ceil(in_shape[2] / self.pool_size[1]),
                    in_shape[3])
        else:
            shape = (in_shape[0],
                    (in_shape[1] // self.pool_size[0]),
                    (in_shape[2] // self.pool_size[1]),
                    in_shape[3])
        return shape

    def compute_padding(self, in_shape):
        mod_y = in_shape[1] % self.pool_size[0]
        y1 = mod_y // 2
        y2 = mod_y - y1
        mod_x = in_shape[2] % self.pool_size[1]
        x1 = mod_x // 2
        x2 = mod_x - x1
        self.padding_shape = ((0, 0), (y1, y2), (x1, x2), (0, 0))

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_shape = self.compute_output_shape(self.in_shape)
        self.compute_padding(self.in_shape)

    def stack(self, inputs):
        if self.padding == "same":
            inputs = tf.pad(inputs, self.padding_shape)
        max_height = (tf.shape(inputs)[1] // self.pool_size[0]) * self.pool_size[0]
        max_width = (tf.shape(inputs)[2] // self.pool_size[1]) * self.pool_size[1]
        stack = tf.stack(
            [inputs[:, i:max_height:self.pool_size[0], j:max_width:self.pool_size[1], :]
             for i in range(self.pool_size[1]) for j in range(self.pool_size[1])],
            axis=-1)
        return stack

    @tf.function
    def call(self, inputs, training=None):
        stacked = self.stack(inputs)
        values = tf.argmax(tf.abs(stacked), axis=-1, output_type=tf.int32)
        stacked_shape = tf.shape(stacked)
        index = tf.stack([
            *tf.meshgrid(
                tf.range(0, stacked_shape[0]),
                tf.range(0, stacked_shape[1]),
                tf.range(0, stacked_shape[2]),
                tf.range(0, stacked_shape[3]),
                indexing='ij'
            ), values],
            axis=-1)

        x = tf.gather_nd(stacked, index)
        x = tf.reshape(x, (-1, *self.out_shape[1:]))
        return x