import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, ELU, \
    Reshape, Softmax, Lambda, Concatenate, Input, Dropout

class ActorModel(object):
    def __init__(self, config):
        self.config = config

        self._input_dim = config.channel_cardinality - 1
        self._output_dim = config.channel_cardinality **2
        self._softmax_dim = config.channel_cardinality

        self.model = self._build_model()

        self.model.summary(line_length=self.config.summary_length), print("\n")

    def _build_model(self):
        model = Sequential(name="Actor")
        for i, hidden in enumerate(self.config.hidden_size):
            kwargs = {'input_shape': [self._input_dim]} if  i == 0 else {}
            model.add(Dense(hidden, activation="elu", name="dense_{}".format(i), **kwargs))
            model.add(Dropout(0.1, name="drop_{}".format(i)))
            # model.add(BatchNormalization(name="BN_{}".format(i)))
            # model.add(ELU())
        model.add(Dense(self._output_dim, name="dense_final"))
        model.add(Dropout(self.config.dropout, name="drop_final"))
        model.add(Reshape((self._softmax_dim, self._softmax_dim)))
        model.add(Softmax(axis=1))
        model.add(Reshape([self._output_dim]))
        return model

    def __call__(self, x, training=False):
        return self.model(x, training=training)


class ChannelModel(object):
    def __init__(self, config, P_out, P_state):
        super(ChannelModel, self).__init__()
        self.config = config

        self._P_out = K.constant(P_out, dtype=tf.float32)
        self._P_state = K.constant(P_state, dtype=tf.float32)

        self._z_dim = config.channel_cardinality - 1
        self._u_dim = config.channel_cardinality


        self.model = self._build_model()
        self.model.summary(line_length=self.config.summary_length), print("\n")

    def _build_model(self):
        inputs = Input(shape=[self._z_dim + self._u_dim ** 2], name="input")
        z_input, u_input = Lambda(self.split, name="split_z_u", input_shape=[self._z_dim + self._u_dim ** 2])(inputs)
        joint = Lambda(self.compute_joint, name="joint")([z_input, u_input])
        reward = Lambda(self._compute_reward, name="reward")(joint)
        z_prime = Lambda(self._compute_next_states, name="next_states")(joint)
        return Model(inputs=[inputs], outputs=[reward, z_prime], name="Channel")

    @tf.function
    def compute_joint(self, zu):
        z,u = zu
        sum_z = K.sum(z, axis=1, keepdims=True)
        z = K.concatenate([z, K.ones_like(sum_z)-sum_z], axis=1)
        z = K.expand_dims(z, axis=1)
        z = K.reshape(z, [-1, 1, self._z_dim+1, 1, 1])

        u = K.reshape(u, [-1, self._u_dim, self._u_dim, 1, 1])

        p_o = K.reshape(self._P_out, [1, self._P_out.shape[0], self._P_out.shape[1], self._P_out.shape[2], 1])
        p_s = K.expand_dims(self._P_state, axis=0)

        joint = z * u * p_o * p_s
        return joint

    @tf.function
    def _compute_reward(self, joint):
        joint = tf.clip_by_value(joint, 0.0, 1.0)
        eps = tf.constant(1e-10)
        p_y = K.sum(joint, axis=(1, 2, 4))
        p_xsy = K.sum(joint, axis=4)


        py_arg = tf.where(tf.greater(p_y, tf.zeros_like(p_y)+eps),
                          -p_y * tf.math.log(p_y+eps) / tf.math.log(2.),
                          tf.zeros_like(p_y))
        pxsy_arg = tf.where(tf.greater(self._P_out, tf.zeros_like(self._P_out)+eps),
                            -p_xsy * tf.math.log(self._P_out+eps) / tf.math.log(2.),
                            tf.zeros_like(p_xsy))
        # pxsy_arg = -p_xsy * tf.math.log(self._P_out+eps) / K.log(2.)
        # pxsy_arg = tf.where(tf.math.is_nan(pxsy_arg), K.zeros_like(pxsy_arg), pxsy_arg)
        reward = K.sum(py_arg, axis=1) - K.sum(pxsy_arg, axis=(1, 2, 3))
        return reward[:, tf.newaxis]

    @tf.function
    def _compute_next_states(self, joint):
            size = K.cast(tf.shape(joint), dtype=tf.int64)[0]
            p_y =  K.sum(joint, axis=(1, 2, 4))

            disturbance = K.squeeze(tf.random.categorical(tf.math.log(p_y), 1), axis=-1)

            next_states = K.sum(joint, axis=(1, 2), keepdims=True) / K.sum(joint, axis=(1, 2, 4), keepdims=True)
            next_states = K.reshape(next_states, shape=[-1, self.output_cardin, self.state_cardin])
            next_states = next_states[:,:,:-1]

            next_state_indices = K.stack([tf.range(size), disturbance], axis=1)
            z_prime = tf.gather_nd(next_states, next_state_indices)
            return z_prime

    # @tf.function
    def split(self, x):
        return tf.split(x, axis=-1, num_or_size_splits=[self._z_dim, self._u_dim ** 2])

    def __call__(self, x):
        return self.model(x)

class IsingModel(ChannelModel):
    def __init__(self, config):
        def f_s(x, s, y, s_prime):
            if x == s_prime:
                p_s_prime = 1.0
            else:
                p_s_prime = 0.0
            return p_s_prime

        def f_y(x, s, y):
            if x == s:
                p_y = (x == y) * 1
            elif x == y:
                p_y = 0.5
            elif s == y:
                p_y = 0.5
            else:
                p_y = 0.
            return p_y

        self.state_cardin = config.channel_cardinality
        self.input_cardin = config.channel_cardinality
        self.output_cardin = config.channel_cardinality

        P_out = np.array([[[f_y(x, s, y) for y in range(self.output_cardin)]
                           for s in range(self.state_cardin)]
                          for x in range(self.input_cardin)])

        P_state = np.array([[[[f_s(x, s, y, s_prime) for s_prime in range(self.state_cardin)]
                              for y in range(self.output_cardin)]
                             for s in range(self.state_cardin)]
                            for x in range(self.input_cardin)])

        super(IsingModel, self).__init__(config, P_out, P_state)
