import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, ELU, \
    Reshape, Softmax, Lambda, Concatenate, Input, Dropout


class ActorModel(object):
    def __init__(self, config, env):
        self.config = config

        self._input_dim = env.z_dim
        self._output_dim = env.u_dim ** 2
        self._softmax_dim = env.u_dim
        print(self._input_dim, self._output_dim, self._softmax_dim)
        self.model = self._build_model()

        self.model.summary(line_length=self.config.summary_length), print("\n")

    def _build_model(self):
        model = Sequential(name="Actor")
        for i, hidden in enumerate(self.config.hidden_size):
            kwargs = {'input_shape': [self._input_dim]} if  i == 0 else {}
            model.add(Dense(hidden, activation='elu', name="dense_{}".format(i), **kwargs))
            # model.add(Dropout(0.0, name="drop_{}".format(i)))
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

        self._z_dim = P_out.shape[1] - 1
        self._u_dim = P_out.shape[0]

        self.model = self._build_model()
        self.model.summary(line_length=self.config.summary_length), print("\n")

    def _build_model(self):
        inputs = Input(shape=[self._z_dim + self._u_dim ** 2], name="input")
        z_input, u_input = Lambda(self.split, name="split_z_u", input_shape=[self._z_dim + self._u_dim ** 2])(inputs)
        joint = Lambda(self.compute_joint, name="joint")([z_input, u_input])
        # disturbance = Lambda(self.compute_disturnabce, name="dist")(joint)
        reward = Lambda(self._compute_reward, name="reward")(joint)
        z_prime, disturbance = Lambda(self._compute_next_states, name="next_states")(joint)
        return Model(inputs=[inputs], outputs=[reward, z_prime, disturbance], name="Channel")

    # @tf.function
    def compute_joint(self, zu):
        z, u = zu
        sum_z = K.sum(z, axis=-1, keepdims=True)
        z = K.concatenate([z, K.ones_like(sum_z)-sum_z], axis=1)
        z = K.expand_dims(z, axis=1)
        z = K.reshape(z, [-1, 1, self._z_dim+1, 1, 1])

        u = tf.reshape(u, [-1, self.u_dim, self.u_dim])
        u = self.constraint_u(u)
        u = K.reshape(u, [-1, self._u_dim, self._u_dim, 1, 1])

        p_o = K.reshape(self._P_out, [1, self._P_out.shape[0], self._P_out.shape[1], self._P_out.shape[2], 1])
        p_s = K.expand_dims(self._P_state, axis=0)

        joint = z * u * p_o * p_s
        # joint = tf.clip_by_value(joint, 1e-6, 1.0)
        return joint

    # def compute_disturnabce(self, joint):
    #     joint = tf.clip_by_value(joint, 1e-6, 1.0)
    #     p_y = K.sum(joint, axis=(1, 2, 4))
    #     return p_y

    # @tf.function
    def _compute_reward(self, joint):
        joint = tf.clip_by_value(joint, 1e-6, 1.0)
        eps = tf.constant(1e-3)
        p_y = K.sum(joint, axis=(1, 2, 4))
        p_xsy = K.sum(joint, axis=4)


        py_arg = tf.where(tf.greater(p_y, tf.zeros_like(p_y)+eps),
                          -p_y * tf.math.log(p_y) / tf.math.log(2.),
                          tf.zeros_like(p_y))
        pxsy_arg = tf.where(tf.greater(self._P_out, tf.zeros_like(self._P_out)+eps),
                            -p_xsy * tf.math.log(self._P_out) / tf.math.log(2.),
                            tf.zeros_like(p_xsy))
        # pxsy_arg = -p_xsy * tf.math.log(self._P_out+eps) / K.log(2.)
        # pxsy_arg = tf.where(tf.math.is_nan(pxsy_arg), K.zeros_like(pxsy_arg), pxsy_arg)
        reward = K.sum(py_arg, axis=1) - K.sum(pxsy_arg, axis=(1, 2, 3))
        return reward[:, tf.newaxis]

    # @tf.function
    def _compute_next_states(self, joint):
        size = K.cast(tf.shape(joint), dtype=tf.int64)[0]
        p_y = K.sum(joint, axis=(1, 2, 4))
        p_y = tf.clip_by_value(p_y, 1e-6, 1.0)
        p_y = p_y / tf.reduce_sum(p_y, axis=-1, keepdims=True)
        disturbance = K.squeeze(tf.random.categorical(tf.math.log(p_y), 1), axis=-1)
        # tf.print(p_y)
        # tf.print(disturbance)

        next_states = K.sum(joint, axis=(1, 2), keepdims=True) / K.sum(joint, axis=(1, 2, 4), keepdims=True)
        next_states = K.reshape(next_states, shape=[-1, self.output_cardin, self.state_cardin])
        shape = tf.shape(next_states)
        next_states = tf.slice(next_states, [0, 0, 0], [shape[0], shape[1], self.z_dim])
        # tf.print(next_states.shape)

        next_state_indices = K.stack([tf.range(size), disturbance], axis=1)
        # tf.print(next_state_indices.shape)
        # tf.print(tf.reduce_max(disturbance), tf.print(p_y))

        z_prime = tf.gather_nd(next_states, next_state_indices)
        disturbance_prob = tf.gather_nd(p_y, next_state_indices)
        return z_prime, disturbance_prob

    @tf.function
    def split(self, x):
        return tf.split(x, axis=-1, num_or_size_splits=[self._z_dim, self._u_dim ** 2])

    @tf.function
    def constraint_u(self, u):
        return u

    def __call__(self, x):
        return self.model(x)

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def u_dim(self):
        return self._u_dim


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


class TrapdoorModel(ChannelModel):
    def __init__(self, config):
        def f_s(x, s, y, s_prime):
            if x == y:
                if s == s_prime:
                    p_s_prime = 1.0
                else:
                    p_s_prime = 0.0
            elif s == y:
                if x == s_prime:
                    p_s_prime = 1.0
                else:
                    p_s_prime = 0.0
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

        super(TrapdoorModel, self).__init__(config, P_out, P_state)


class RLL_0_1(ChannelModel):
    def __init__(self, config, eps=0.0):
        def f_s(x, s, y, s_prime):
            if s == self.cardinality-1 or x == 1:
                if s_prime == 0:
                    p_s_prime = 1.0
                else:
                    p_s_prime = 0.0
            else:
                if s+1 == s_prime:
                    p_s_prime = 1.0
                else:
                    p_s_prime = 0.0

            return p_s_prime

        def f_y(x,s,y):
                if y == 2:
                    p_y = self.eps
                elif x == y:
                    p_y = 1-self.eps
                else:
                    p_y = 0.0
                return p_y

        self.cardinality = 2
        self.state_cardin = 2
        self.input_cardin = 2
        self.output_cardin = 3
        self.eps = eps
        P_out = np.array([[[f_y(x, s, y) for y in range(self.output_cardin)]
                                       for s in range(self.state_cardin)]
                                       for x in range(self.input_cardin)])

        P_state = np.array([[[[ f_s(x, s, y, s_prime)   for s_prime in range(self.state_cardin)]
                                                                    for y in range(self.output_cardin)]
                                                                    for s in range(self.state_cardin)]
                                                                    for x in range(self.input_cardin)])

        print(P_out.shape, P_state.shape)
        super(RLL_0_1, self).__init__(config, P_out, P_state)

    # @tf.function
    def constraint_u(self, u):
        # shape = tf.shape(u)
        mask = tf.expand_dims(tf.constant([[1, 1], [1, 0]], dtype=tf.float32), axis=0)
        # mask = tf.tile(mask_, [shape[0], 1, 1])
        u_ = u * mask
        u_ += 1e-6
        u_constraint = u_ / tf.reduce_sum(u_, axis=1, keepdims=True)
    #     u = u * mask
    #     u_constraint = u / tf.reduce_sum(u, axis=-1, keepdims=True)
        return u_constraint