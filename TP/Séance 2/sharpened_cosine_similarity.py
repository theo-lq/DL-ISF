import math

import tensorflow as tf


class CosSim2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        stride=1,
        padding="valid",
        kernel_initializer="glorot_uniform",
        depthwise_separable=False,
        **kwargs
    ):

        super(CosSim2D, self).__init__(**kwargs)
        self.filters = filters
        assert kernel_size in [1, 3, 5], "kernel of this size not supported"
        self.kernel_size = kernel_size
        if self.kernel_size == 1:
            self.stack = lambda x: x
        elif self.kernel_size == 3:
            self.stack = self.stack3x3
        elif self.kernel_size == 5:
            self.stack = self.stack5x5
        self.stride = stride
        if padding == "same":
            self.pad = self.kernel_size // 2
            self.pad_1 = 1
            self.clip = 0
        elif padding == "valid":
            self.pad = 0
            self.pad_1 = 0
            self.clip = self.kernel_size // 2
        self.kernel_initializer = kernel_initializer
        self.depthwise_separable = depthwise_separable

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_y = math.ceil((self.in_shape[1] - 2 * self.clip) / self.stride)
        self.out_x = math.ceil((self.in_shape[2] - 2 * self.clip) / self.stride)
        self.flat_size = self.out_x * self.out_y
        self.channels = self.in_shape[3]

        if self.depthwise_separable:
            self.w = self.add_weight(
                shape=(1, tf.square(self.kernel_size), self.filters),
                initializer=self.kernel_initializer, name='w',
                trainable=True,
            )
        else:
            self.w = self.add_weight(
                shape=(1, self.channels * tf.square(self.kernel_size), self.filters),
                initializer=self.kernel_initializer, name='w',
                trainable=True,
            )

        p_init = tf.constant_initializer(value=100**0.5)
        self.p = self.add_weight(shape=(self.filters,), initializer=p_init, trainable=True, name='p')

        q_init = tf.constant_initializer(value=10**0.5)
        self.q = self.add_weight(shape=(1,), initializer=q_init, trainable=True, name='q')

    def l2_normal(self, x, axis=None, epsilon=1e-12):
        square_sum = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        x_inv_norm = tf.sqrt(tf.maximum(square_sum, epsilon))
        return x_inv_norm

    def stack3x3(self, image):
        '''
            sliding window implementation for 3x3 kernel
        '''
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack(
            [
                tf.pad(  # top row
                    image[:, :y - 1 - self.clip:, :x - 1 - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 1 - self.clip, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 1 - self.clip, 1 + self.clip:, :],
                    tf.constant([[0, 0], [self.pad, 0], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # middle row
                    image[:, self.clip:y - self.clip, :x - 1 - self.clip, :],
                    tf.constant([[0, 0], [0, 0], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                image[:, self.clip:y - self.clip:self.stride, self.clip:x - self.clip:self.stride, :],
                tf.pad(
                    image[:, self.clip:y - self.clip, 1 + self.clip:, :],
                    tf.constant([[0, 0], [0, 0], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # bottom row
                    image[:, 1 + self.clip:, :x - 1 - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:, 1 + self.clip:, :],
                    tf.constant([[0, 0], [0, self.pad], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :]
            ], axis=3)
        return stack

    def stack5x5(self, image):
        '''
            sliding window implementation for 5x5 kernel
        '''
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack(
            [
                tf.pad(  # top row
                    image[:, :y - 2 - self.clip:, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 2 - self.clip:, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 2 - self.clip:, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 2 - self.clip:, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [self.pad, 0], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 2 - self.clip:, 2 + self.clip:, :],
                    tf.constant([[0, 0], [self.pad, 0], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # 2nd row
                    image[:, 1:y - 1 - self.clip:, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1:y - 1 - self.clip:, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1:y - 1 - self.clip:, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1:y - 1 - self.clip:, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1:y - 1 - self.clip:, 2 + self.clip:, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # 3rd row
                    image[:, self.clip:y - self.clip, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [0, 0], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, self.clip:y - self.clip, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [0, 0], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                image[:, self.clip:y - self.clip, self.clip:x - self.clip, :][:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, self.clip:y - self.clip, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [0, 0], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, self.clip:y - self.clip, 2 + self.clip:, :],
                    tf.constant([[0, 0], [0, 0], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # 4th row
                    image[:, 1 + self.clip:-1, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:-1, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:-1, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:-1, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:-1, 2 + self.clip:, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # 5th row
                    image[:, 2 + self.clip:, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 2 + self.clip:, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 2 + self.clip:, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 2 + self.clip:, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [0, self.pad], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 2 + self.clip:, 2 + self.clip:, :],
                    tf.constant([[0, 0], [0, self.pad], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
            ], axis=3)
        return stack

    def call_body(self, inputs):
        channels = tf.shape(inputs)[-1]
        x = self.stack(inputs)
        x = tf.reshape(x, (-1, self.flat_size, channels * tf.square(self.kernel_size)))
        x_norm = (self.l2_normal(x, axis=2) + tf.square(self.q)/10)
        w_norm = (self.l2_normal(self.w, axis=1) + tf.square(self.q)/10)
        x = tf.matmul(x / x_norm, self.w / w_norm)
        sign = tf.sign(x)
        x = tf.abs(x) + 1e-12
        x = tf.pow(x, tf.square(self.p)/100)
        x = sign * x
        x = tf.reshape(x, (-1, self.out_y, self.out_x, self.filters))
        return x

    @tf.function
    def call(self, inputs, training=None):
        if self.depthwise_separable:
            x = tf.vectorized_map(self.call_body, tf.expand_dims(tf.transpose(inputs, (3, 0, 1, 2)), axis=-1),
                                  fallback_to_while_loop=True)
            x = tf.transpose(x, (1, 2, 3, 4, 0))
            x = tf.reshape(x, (-1, self.out_y, self.out_x, self.channels * self.filters))
        else:
            x = self.call_body(inputs)
        return x
