import os
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
from sklearn.cluster import KMeans
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, ELU, \
    Reshape, Softmax, Lambda, Concatenate, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Mean
import matplotlib.pyplot as plt


def loss(y_true, y_pred):
    return -K.mean(y_pred)

class POTrainer:
    def __init__(self, actor, env, config):
        self.actor = actor
        self.env = env

        self.config = config
        self.T = config.unroll_steps
        self.input_dim = env.z_dim
        self.model = self._build_training_model()

        lr_sche = keras.optimizers.schedules.ExponentialDecay(self.config.learning_rate,
                                                              decay_steps=self.config.num_epochs // self.config.learning_rate_decay_steps,
                                                              decay_rate=self.config.learning_rate_decay,
                                                              staircase=False)
        self.optimizer = Adam(learning_rate=lr_sche)

        self.model.compile(optimizer=self.optimizer,
                           loss=loss)

        self.mean_metric = Mean()


    @tf.function
    def split_rewards_and_state(self, x):
        return tf.split(x, axis=-1, num_or_size_splits=[1, self.input_dim])

    def _build_training_model(self):
        def name(title, idx):
            return "{}_{:d}".format(title, idx)
        rewards = list()
        states = list()
        disturbances = list()
        self.input = z = Input(shape=[self.input_dim])
        for t in range(self.T):
            u = self.actor.model(z, training=True)
            z_u = Concatenate(axis=-1, name=name("concat_z_u", t))([z, u])
            r, z_prime, p_y = self.env.model(z_u)

            rewards.append(r)
            z = z_prime
            states.append(z)
            disturbances.append(p_y)

        self.rewards = Concatenate(axis=-1, name="concat_rewards")(rewards)
        self.states = Lambda(tf.stack, arguments={'axis': 1, 'name': "concat_states"})(states)
        self.disturbances = Lambda(tf.stack, arguments={'axis': 1, 'name': "concat_dist"})(disturbances)

        return Model(inputs=self.input, outputs=[self.rewards, self.states, self.disturbances])

    @tf.function
    def train_epoch(self):
        raise NotImplementedError

    # @tf.function
    def train_step(self, z):
        with tf.GradientTape(persistent=True) as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            rewards, states, disturbances = self.model(z)
            # downstream_reward_average = tf.math.cumsum(tf.stop_gradient(rewards), axis=1, reverse=True) / \
            #                             tf.tile(tf.reshape(tf.range(tf.shape(rewards)[1], 0, -1, tf.float32), [1, -1]), [tf.shape(rewards)[0], 1])
            downstream_reward_average = tf.math.cumsum(tf.stop_gradient(rewards - K.mean(rewards)), axis=1, reverse=True)
            loss1 = -K.mean(rewards - tf.stop_gradient(K.mean(rewards)))
            loss2 = -K.mean(tf.math.log(disturbances) * downstream_reward_average)
            # loss2 = -K.mean(K.sum(tf.math.log(disturbances),axis=1) * tf.stop_gradient(rewards[:, -1]))
            # loss2 = -K.mean(tf.math.log(disturbances) * tf.math.cumsum(tf.stop_gradient(rewards), axis=1, reverse=True))
            loss = loss1 + loss2
        gradients1 = tape.gradient(loss1, self.actor.model.trainable_weights)
        gradients2 = tape.gradient(loss2, self.actor.model.trainable_weights)
        tf.print(tf.linalg.global_norm(gradients1), tf.linalg.global_norm(gradients2))
        gradients = [g1+g2 for g1, g2 in zip(gradients1, gradients2)]
        # gradients = tape.gradient(loss, self.actor.model.trainable_weights)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 0.1)
        self.optimizer.apply_gradients(zip(gradients, self.actor.model.trainable_weights))

        return -loss, states[:,-1,:]
        # train_loss(loss)
        # train_accuracy(labels, predictions)

    def train(self):
        def lr(k):
            return self.config.learning_rate*(self.config.learning_rate_decay**
                                              (k//(self.config.num_epochs//self.config.learning_rate_decay_steps)))

        template = "Epoch: {:05d}\tLearning Rate: {:2.2e}\tAverage Reward: {:8.5f} "

        z = tf.random.uniform([self.config.batch_size, self.input_dim])
        for k in range(self.config.num_epochs):

            I, z = self.train_step(z)
            if k % self.config.eval_freq == 0:
                average_reward, state_histogram = self.test(self.config.eval_len)
                plt.hist(np.squeeze(state_histogram), bins=50)
                plt.savefig(os.path.join(self.config.summary_dir, "hist-{}.png".format(k)))
                plt.close()
                print(template.format(k, lr(k), average_reward))
                # print("average_reward", float(average_reward), k)

        average_reward, state_histogram = self.test(self.config.eval_long_len)
        # print("average_reward", float(average_reward), self.config.num_epochs)
        print("Epoch: {}\tAverage Reward: {:8.5f} ".format("Final", average_reward))
        state_clusters = KMeans(n_clusters=self.config.n_clusters).fit(state_histogram)
        print(['{}\n'.format(x) for x in state_clusters.cluster_centers_])

        print(self.config.summary_dir)
        with open(os.path.join(self.config.experiment_dir, "log.txt"), 'a') as f:
            f.write("average_reward: {}\n".format(average_reward))
            f.write("Clusters:\n")
            f.writelines(['{}\n'.format(x) for x in state_clusters.cluster_centers_])
        with open(os.path.join(os.path.dirname(self.config.experiment_dir), "log.txt"), 'a') as f:
            f.write("{}\n".format(average_reward))
        print(*['{}\n'.format(x) for x in state_clusters.cluster_centers_])

    def test(self, eval_len):
        state_histogram = list()
        self.mean_metric.reset_states()
        z = tf.zeros([self.config.batch_size_eval, self.input_dim])
        for k in tqdm(range(eval_len)):
            r, next_states, disturbances = self.model.predict(z)
            z = tf.squeeze(next_states[:,-1,:])
            if k * self.config.unroll_steps > 100:
                self.mean_metric(r)
                state_histogram.append(next_states)

        return self.mean_metric.result(), tf.reshape(tf.concat(state_histogram, axis=0), [-1, self.input_dim])