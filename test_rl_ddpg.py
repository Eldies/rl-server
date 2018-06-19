import numpy as np
import tempfile
import tensorflow as tf
import threading

from tf_rl.controller import ContinuousDeepQ
from tf_rl.models import MLP

from tf_rl.utils import base_name2

from tensorflow.python.ops import array_ops

class HEAD(object):
    def __init__(self,
                 input_size,
                 step_count=5,
                 layer_size=32,
                 layers_count=2,
                 output_size=8,
                 batch_size=32,
                 mlp=None,
                 scope='lstm_mlp',
                 train_loop=None,
                 activation=tf.nn.tanh,
                 ):
        self.input_size = input_size
        self.step_count = step_count
        self.layer_size = layer_size
        self.layers_count = layers_count
        self.batch_size = batch_size
        self.mlp = mlp
        self.scope = scope
        self.train_loop = train_loop
        self.activation = activation
        self.output_size = output_size

        with tf.variable_scope(scope) as sc:
            def lstm_cell():
                # return tf.contrib.rnn.BasicLSTMCell(self.layer_size, reuse=sc.reuse)
                return tf.contrib.rnn.LSTMCell(
                    self.layer_size,
                    reuse=sc.reuse,
                    # initializer=tf.random_uniform_initializer(-0.05, 0.05)
                    activation=self.activation,
                )

            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.layers_count)])
            if self.output_size is not None:
                self.output_layer = tf.layers.Dense(
                    self.output_size,
                    _reuse=sc.reuse,
                    activation=None,
                )
            else:
                self.output_layer = lambda x: x

            print([self.batch_size, self.step_count, self.input_size])
            fake_input = tf.placeholder(tf.float32, [self.batch_size, self.step_count, self.input_size])

            self.initial_state_batch = self.stacked_lstm.zero_state(self.train_loop.batch_size, tf.float32)
            self.initial_state_one = self.stacked_lstm.zero_state(1, tf.float32)
            self.lstm_output, state = self.stacked_lstm(fake_input[:, 0], self.initial_state_batch)
            self.output = self.output_layer(self.lstm_output)

            self.model_variables = [v for v in tf.trainable_variables() if v.name.startswith(sc.name)]
            for v in self.model_variables:
                print ("--- MULTI_LSTM_MLP v: " + v.name)

    def __call__(self, xs):
        # if this is critic we need to ignore input action
        # since it is already present
        print ('call: ' + (self.scope if isinstance(self.scope, str) else self.scope.name))
        if(isinstance(xs, list)):
            lstm_input = xs[0]
        else:
            lstm_input = xs
        print (lstm_input)

        # convert xs into steps
        lstm_input = tf.reshape(lstm_input, [-1, self.step_count, self.input_size])
        print (lstm_input.get_shape())

        initial_state = self.initial_state_batch
        if str(lstm_input.get_shape()[0]) == '?':
            print('--- dynamic shape')
            initial_state = self.initial_state_one
        print('--- initial state')
        print(lstm_input.get_shape()[0])

        with tf.variable_scope(self.scope, reuse=True):
            state = initial_state
            for i in range(self.step_count):
                print ('--- lstm step: {}'.format(i))
                print (lstm_input[:, i].get_shape())
                lstm_output, state = self.stacked_lstm(lstm_input[:, i], state)
            final_state = state

        if str(lstm_input.get_shape()[0]) == '?':
            mlp_input = array_ops.concat([self.output_layer(lstm_output), lstm_input[:1, -1, :4]], 1)
        else:
            mlp_input = array_ops.concat([self.output_layer(lstm_output), lstm_input[:, -1, :4]], 1)

        if(isinstance(xs, list)):
            return self.mlp([mlp_input, xs[1]])
        else:
            return self.mlp(mlp_input)

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        print ("--- copy " + scope)
        with tf.variable_scope(scope) as sc:
            for v in self.model_variables:
                print ("--- bn: " + base_name2(v) + " " + v.name)
                tf.get_variable(base_name2(v), v.get_shape(), initializer=lambda x,dtype=tf.float32, partition_info=None: v.initialized_value())
            sc.reuse_variables()
        mlp_copy = self.mlp.copy('mlp_'+scope)
        return HEAD(
            self.input_size,
            self.step_count,
            self.layer_size,
            self.layers_count,
            self.output_size,
            self.batch_size,
            mlp_copy,
            scope=sc,
            train_loop=self.train_loop,
            activation=self.activation,
        )

    def variables(self):
        return self.model_variables + self.mlp.variables()


class TestRL ():

    def __init__ (self, train_loop):

        self.train_loop = train_loop
        self.graph = train_loop.graph
        self.sess = train_loop.sess
        journalist = train_loop.logger

        num_actions = self.train_loop.num_actions;
        observation_size = self.train_loop.observation_size;
        observations_in_seq = 1;
        input_size = observation_size*observations_in_seq;
        learning_rate = 1e-4

        lr = tf.nn.leaky_relu
        r = tf.nn.relu
        t = tf.nn.tanh

        self.lstm_input_size = 6
        self.lstm_layer_size = 8
        self.lstm_layers_count = 1
        self.lstm_steps_count = 1
        self.head_output_size = 8

        USE_HEAD = False
        mlp_input_size = self.head_output_size + 4 if USE_HEAD else input_size

        mlp_critic = MLP([mlp_input_size, num_actions], [512, 512, 512, 1],
                    [lr, lr, lr, tf.identity], scope='critic')

        mlp_actor = MLP([mlp_input_size,], [512, 512, 512, num_actions],
                    [lr, lr, lr, t], scope='actor')

        if USE_HEAD:
            self.actor = HEAD(
                self.lstm_input_size,
                self.lstm_steps_count,
                self.lstm_layer_size,
                self.lstm_layers_count,
                #self.head_output_size,
                None,
                train_loop.batch_size,
                mlp_actor,
                'actor',
                train_loop=train_loop,
    #            activation=lr,
            )
            critic = HEAD(
                self.lstm_input_size,
                self.lstm_steps_count,
                self.lstm_layer_size,
                self.lstm_layers_count,
                #self.head_output_size,
                None,
                train_loop.batch_size,
                mlp_critic,
                'critic',
                train_loop=train_loop,
#                activation=lr,
            )
        else:
            self.actor = mlp_actor
            critic = mlp_critic

        # step 1
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # step 2
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-5)

        self.controller = ContinuousDeepQ(
            input_size,
            num_actions,
            self.actor,
            critic,
            optimizer,
            self.sess,
#            discount_rate=0.9,
#            target_actor_update_rate=0.01,
#            target_critic_update_rate=0.01,
            exploration_period=5000,
            max_experience=10000,
            store_every_nth=4,
            train_every_nth=4,
            summary_writer=journalist,
            rewards = self.train_loop.dequeued_rewards,
            given_action = self.train_loop.dequeued_actions,
            observation = self.train_loop.dequeued_prev_states,
            next_observation = self.train_loop.dequeued_next_states,
            next_observation_mask = tf.ones(self.train_loop.dequeued_rewards.get_shape (), tf.float32)
        )

    def act (self, state):
        [a] = self.sess.run (
            self.controller.actor_val,
            {
                self.controller.observation_for_act: np.array(state).reshape((1,-1))
            }
        )
        return a.tolist ()

    def act_batch (self, states):
        a = self.sess.run (
            self.controller.actor_val,
            {
                self.controller.observation_for_act: np.array(states)
            }
        )
        return a.tolist ()

    def get_loss_op (self):
        return self.controller.critic_error

    def get_train_ops (self):
        return [
            self.controller.actor_update,
            self.controller.critic_update,
            self.controller.update_all_targets
        ]
