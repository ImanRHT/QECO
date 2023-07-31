import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate
from tensorflow.keras.optimizers import RMSprop

class DuelingDoubleDeepQNetwork:

    def __init__(self,
                 n_actions,                  # the number of actions
                 n_features,
                 n_lstm_features,
                 n_time,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.99,
                 replace_target_iter=200,    # each 200 steps, update target net
                 memory_size=500,            # maximum of memory
                 batch_size=32,
                 e_greedy_increment=0.00025,
                 n_lstm_step=10,
                 dueling=True,
                 double_q=True,
                 hidden_units_l1=20,
                 N_lstm=20):

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.hidden_units_l1 = hidden_units_l1

        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_features

        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                + self.n_features + self.n_lstm_state + self.n_lstm_state))

        self._build_net()

        t_params = self.target_net.get_weights()
        self.eval_net.set_weights(t_params)

        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = list()

    def _build_net(self):

        def build_layers(s, lstm_s, hidden_units_l1, n_lstm):
            lstm_input = Input(shape=(self.n_lstm_step, self.n_lstm_state))
            lstm_output = LSTM(n_lstm, return_sequences=False)(lstm_input)
            lstm_model = Model(inputs=lstm_input, outputs=lstm_output)

            input_layer = Input(shape=(self.n_features,))
            concat_layer = Concatenate(axis=-1)([lstm_model(lstm_s), input_layer])

            l1 = Dense(hidden_units_l1, activation='relu')(concat_layer)
            l12 = Dense(hidden_units_l1, activation='relu')(l1)

            if self.dueling:
                value = Dense(1, activation='linear')(l12)
                advantage = Dense(self.n_actions, activation='linear')(l12)
                advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
                q_values = value + (advantage - advantage_mean)
            else:
                q_values = Dense(self.n_actions, activation='linear')(l1)

            model = Model(inputs=[input_layer, lstm_input], outputs=q_values)
            return model

        input_s = Input(shape=(self.n_features,))
        input_lstm_s = Input(shape=(self.n_lstm_step, self.n_lstm_state))

        self.eval_net = build_layers(input_s, input_lstm_s, self.hidden_units_l1, self.N_lstm)
        self.target_net = build_layers(input_s, input_lstm_s, self.hidden_units_l1, self.N_lstm)

        self.target_net.set_weights(self.eval_net.get_weights())

        self.eval_net.compile(loss='mean_squared_error', optimizer=RMSprop(lr=self.lr))

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            lstm_observation = np.array(self.lstm_history)
            actions_value = self.eval_net.predict([observation, lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)])
            self.store_q_value.append({'observation': observation, 'q_value': actions_value})
            action = np.argmax(actions_value)
        else:
            if np.random.randint(0, 100) < 25:
                action = np.random.randint(1, self.n_actions)
            else:
                action = 0

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            t_params = self.target_net.get_weights()
            self.eval_net.set_weights(t_params)
            print('\ntarget_params_replaced')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

        batch_memory = self.memory[sample_index, :self.n_features + 1 + 1 + self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])
        for i in range(len(sample_index)):
            for j in range(self.n_lstm_step):
                lstm_batch_memory[i, j, :] = self.memory[sample_index[i] + j, self.n_features + 1 + 1 + self.n_features:]

        q_next, q_eval4next = self.target_net.predict([batch_memory[:, -self.n_features:], lstm_batch_memory[:, :, self.n_lstm_state:]])
        q_eval = self.eval_net.predict([batch_memory[:, :self.n_features], lstm_batch_memory[:, :, :self.n_lstm_state]])

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        loss = self.eval_net.train_on_batch([batch_memory[:, :self.n_features], lstm_batch_memory[:, :, :self.n_lstm_state]], q_target)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self, episode, time, action):
        while episode >= len(self.action_store):
            self.action_store.append(-np.ones([self.n_time]))
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

    def Initialize(self, sess, iot):
        self.sess = sess
        self.load_model(iot)

    def load_model(self, iot):
        latest_ckpt = tf.train.latest_checkpoint("./models/500/" + str(iot) + "_X_model")
        print(latest_ckpt, "_____+______________________________________________")
        if latest_ckpt is not None:
            self.eval_net.load_weights(latest_ckpt)
