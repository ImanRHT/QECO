import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class DuelingDoubleDeepQNetwork:

    def __init__(self,
                 n_actions,               
                 n_features,
                 n_lstm_features,
                 n_time,
                 learning_rate = 0.01,
                 reward_decay = 0.9,
                 e_greedy = 0.99,
                 replace_target_iter = 200,  
                 memory_size = 500,  
                 batch_size=32,
                 e_greedy_increment= 0.00025,
                 n_lstm_step = 10,
                 dueling = True,
                 double_q = True,
                 N_L1 = 20,
                 N_lstm = 20):

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size    # select self.batch_size number of time sequence for learning
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.N_L1 = N_L1

        # lstm
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step       # step_size in lstm
        self.n_lstm_state = n_lstm_features  # [fog1, fog2, ...., fogn, M_n(t)]

        # initialize zero memory np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                    + self.n_features + self.n_lstm_state + self.n_lstm_state))

        # consist of [target_net, evaluate_net]
        self._build_net()

        # replace the parameters in target net
        t_params = tf.get_collection('target_net_params')  # obtain the parameters in target_net
        e_params = tf.get_collection('eval_net_params')  # obtain the parameters in eval_net
        self.replace_target_op = [tf.assign(t, e) for t, e in
                                      zip(t_params, e_params)]  # update the parameters in target_net

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for ii in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = list()

        self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def _build_net(self):

        tf.reset_default_graph()

        def build_layers(s,lstm_s,c_names, n_l1, n_lstm, w_initializer, b_initializer):

            # lstm for load levels
            with tf.variable_scope('l0'):
                lstm_dnn = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm)
                lstm_dnn.zero_state(self.batch_size, tf.float32)
                lstm_output,lstm_state = tf.nn.dynamic_rnn(lstm_dnn, lstm_s, dtype=tf.float32)
                lstm_output_reduced = tf.reshape(lstm_output[:, -1, :], shape=[-1, n_lstm])

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',[n_lstm + self.n_features, n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1',[1,n_l1],initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(tf.concat([lstm_output_reduced, s],1), w1) + b1)

            # second layer
            with tf.variable_scope('l12'):
                w12 = tf.get_variable('w12', [n_l1, n_l1], initializer=w_initializer,
                                         collections=c_names)
                b12 = tf.get_variable('b12', [1, n_l1], initializer=b_initializer, collections=c_names)
                l12 = tf.nn.relu(tf.matmul(l1, w12) + b12)

            # the second layer is different
            if self.dueling:
                # Dueling DQN
                # a single output n_l1 -> 1
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2',[n_l1,1],initializer=w_initializer,collections=c_names)
                    b2 = tf.get_variable('b2',[1,1],initializer=b_initializer,collections=c_names)
                    self.V = tf.matmul(l12,w2) + b2
                # n_l1 -> n_actions
                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
                    b2 = tf.get_variable('b2',[1,self.n_actions],initializer=b_initializer,collections=c_names)
                    self.A = tf.matmul(l12,w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims=True))  # Q = V(s) +A(s,a)

            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        # input for eval_net
        self.s = tf.placeholder(tf.float32,[None,self.n_features], name = 's')  # state (observation)
        self.lstm_s = tf.placeholder(tf.float32,[None,self.n_lstm_step,self.n_lstm_state], name='lstm1_s')

        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions], name = 'Q_target') # q_target

        # input for target_net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.lstm_s_ = tf.placeholder(tf.float32,[None,self.n_lstm_step,self.n_lstm_state], name='lstm1_s_')

        # generate EVAL_NET, update parameters
        with tf.variable_scope('eval_net'):

            # c_names(collections_names), will be used when update target_net
            # tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32), return a initializer
            c_names, n_l1, n_lstm, w_initializer, b_initializer =  \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.N_L1, self.N_lstm,\
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # input (n_feature) -> l1 (n_l1) -> l2 (n_actions)
            self.q_eval = build_layers(self.s, self.lstm_s, c_names, n_l1, n_lstm, w_initializer, b_initializer)

        # generate TARGET_NET
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, self.lstm_s_, c_names, n_l1, n_lstm, w_initializer, b_initializer)

        # loss and train
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, lstm_s,  a, r, s_, lstm_s_):
        # RL.store_transition(observation,action,reward,observation_)
        # hasattr(object, name), if object has name attribute
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # store np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))  # stack in horizontal direction

        # if memory overflows, replace old memory with new one
        index = self.memory_counter % self.memory_size
        # print(transition)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):

        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        # the shape of the observation (1, size_of_observation)
        # x1 = np.array([1, 2, 3, 4, 5]), x1_new = x1[np.newaxis, :], now, the shape of x1_new is (1, 5)
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:

            # lstm only contains history, there is no current observation
            lstm_observation = np.array(self.lstm_history)

            actions_value = self.sess.run(self.q_eval,
                                          feed_dict={self.s: observation,
                                                     self.lstm_s: lstm_observation.reshape(1, self.n_lstm_step,
                                                                                           self.n_lstm_state),
                                                     })

            self.store_q_value.append({'observation': observation, 'q_value': actions_value})

            action = np.argmax(actions_value)

        else:
            if np.random.randint(0,100)>101:
                action = 0
            else:
                action = np.random.randint(1, self.n_actions)   
        return action

    def learn(self):

        # check if replace target_net parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            # run the self.replace_target_op in __int__
            self.sess.run(self.replace_target_op)
            print('Network_parameter_updated\n')

        # randomly pick [batch_size] memory from memory np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)\

        #  transition = np.hstack(s, [a, r], s_, lstm_s, lstm_s_)
        batch_memory = self.memory[sample_index, :self.n_features+1+1+self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii,jj,:] = self.memory[sample_index[ii]+jj,
                                              self.n_features+1+1+self.n_features:]

        # obtain q_next (from target_net) (to q_target) and q_eval (from eval_net)
        # minimize（target_q - q_eval）^2
        # q_target = reward + gamma * q_next
        # in the size of bacth_memory
        # q_next, given the next state from batch, what will be the q_next from q_next
        # q_eval4next, given the next state from batch, what will be the q_eval4next from q_eval
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],  # output
            feed_dict={
                # [s, a, r, s_]
                # input for target_q (last)
                self.s_: batch_memory[:, -self.n_features:], self.lstm_s_: lstm_batch_memory[:,:,self.n_lstm_state:],
                # input for eval_q (last)
                self.s: batch_memory[:, -self.n_features:], self.lstm_s: lstm_batch_memory[:,:,self.n_lstm_state:],
            }
        )
        # q_eval, given the current state from batch, what will be the q_eval from q_eval
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features],
                                                 self.lstm_s: lstm_batch_memory[:,:,:self.n_lstm_state]})
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # action with a single value (int action)
        reward = batch_memory[:, self.n_features + 1]  # reward with a single value

        # update the q_target at the particular batch at the correponding action
        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # both self.s and self.q_target belong to eval_q
        # input self.s and self.q_target, output self._train_op, self.loss (to minimize the gap)
        # self.sess.run: given input (feed), output the required element
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.lstm_s: lstm_batch_memory[:, :, :self.n_lstm_state],
                                                self.q_target: q_target})

        # gradually increase epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return self.cost

    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self,episode,time, action):
        while episode >= len(self.action_store):
            self.action_store.append(- np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay


    def do_store_energy(self, episode, time, energy, energy2, energy3, energy4):
    
        fog_energy = 0
        for i in range(len(energy3)):
            if energy3[i] != 0:
                fog_energy = energy3[i]


        idle_energy = 0
        for i in range(len(energy4)):
            if energy4[i] != 0:
                idle_energy = energy4[i]

        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy + energy2 + fog_energy + idle_energy

    def load_model(self,iot):
        latest_ckpt = tf.train.latest_checkpoint("500-"+str(iot)+"_model")

        print(latest_ckpt, "_____+______________________________________________")
        if latest_ckpt is not None:
            self.saver.restore(self.sess, latest_ckpt)