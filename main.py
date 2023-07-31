from MEC_Env import MEC
from DDQN import DuelingDoubleDeepQNetwork
from Config import Config
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil

'''
if not os.path.exists("models"):
    os.mkdir("models")
else:
    shutil.rmtree("models")
    os.mkdir("models")
'''

def reward_fun(ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy, delay, max_delay, unfinish_task):
    edge_energy  = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)
    penalty     = -max_delay*4
    if unfinish_task == 1:
        reward = penalty
    else:
        reward = 0
    reward = reward - (ue_comp_energy + ue_trans_energy + edge_energy + idle_energy)
    return reward

def monitor_reward(ue_RL_list, episode):
    episode_sum_reward = sum(sum(ue_RL.reward_store[episode]) for ue_RL in ue_RL_list)
    avg_episode_sum_reward = episode_sum_reward / len(ue_RL_list)
    print(f"reward: {avg_episode_sum_reward}")
    return avg_episode_sum_reward

def monitor_delay(ue_RL_list, episode):
    delay_ue_list = [sum(ue_RL.delay_store[episode]) for ue_RL in ue_RL_list]
    avg_delay_in_episode = sum(delay_ue_list) / len(delay_ue_list)
    print(f"delay: {avg_delay_in_episode}")
    return avg_delay_in_episode

def monitor_energy(ue_RL_list, episode):
    energy_ue_list = [sum(ue_RL.energy_store[episode]) for ue_RL in ue_RL_list]
    avg_energy_in_episode = sum(energy_ue_list) / len(energy_ue_list)
    print(f"energy: {avg_energy_in_episode}")
    return avg_energy_in_episode

def cal_reward(ue_RL_list):
    total_sum_reward = 0
    num_episodes = 0
    for ue_num, ue_RL in enumerate(ue_RL_list):
        print("________________________")
        print("ue_num:", ue_num)
        print("________________________")
        for episode, reward in enumerate(ue_RL.reward_store):
            print("episode:", episode)
            reward_sum = sum(reward)
            print(reward_sum)
            total_sum_reward += reward_sum
            num_episodes += 1
    avg_reward = total_sum_reward / num_episodes
    print(total_sum_reward, avg_reward)


def train(ue_RL_list, NUM_EPISODE):
    avg_reward_list = []
    avg_reward_list_2 = []
    avg_delay_list_in_episode = []
    avg_energy_list_in_episode = []
    num_task_drop_list_in_episode = []
    RL_step = 0
    a = 1

    for episode in range(NUM_EPISODE):
        print("episode  :", episode)
        print("epsilon  :", ue_RL_list[0].epsilon)

        # BITRATE ARRIVAL
        bitarrive = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
        task_prob = env.task_arrive_prob
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_ue])

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for ue_index in range(env.n_ue):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_ue])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive)

        # TRAIN DRL
        while True:

            # PERFORM ACTION
            action_all = np.zeros([env.n_ue])
            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :])
                if np.sum(observation) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    action_all[ue_index] = 0
                else:
                    action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation)
                    if observation[0] != 0:
                        ue_RL_list[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # should store this information in EACH time slot
            for ue_index in range(env.n_ue):
                ue_RL_list[ue_index].update_lstm(lstm_state_all_[ue_index,:])

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for ue_index in range(env.n_ue):
                obs = observation_all[ue_index, :]
                lstm = np.squeeze(lstm_state_all[ue_index, :])
                action = action_all[ue_index]
                obs_ = observation_all_[ue_index]
                lstm_ = np.squeeze(lstm_state_all_[ue_index,:])
                history[env.time_count - 1][ue_index].update({
                    'observation': obs,
                    'lstm': lstm,
                    'action': action,
                    'observation_': obs_,
                    'lstm_': lstm_
                })

                update_index = np.where((1 - reward_indicator[:,ue_index]) *env.process_delay[:,ue_index] > 0)[0]
                if len(update_index) != 0:
                    for time_index in update_index:
                        reward = reward_fun(
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy [time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index],
                            env.process_delay[time_index, ue_index],
                            env.max_delay,
                            env.unfinish_task[time_index, ue_index]
                        )
                        ue_RL_list[ue_index].store_transition(
                            history[time_index][ue_index]['observation'],
                            history[time_index][ue_index]['lstm'],
                            history[time_index][ue_index]['action'],
                            reward,
                            history[time_index][ue_index]['observation_'],
                            history[time_index][ue_index]['lstm_']
                        )
                        ue_RL_list[ue_index].do_store_reward(
                            episode,
                            time_index,
                            reward
                        )
                        ue_RL_list[ue_index].do_store_delay(
                            episode,
                            time_index,
                            env.process_delay[time_index, ue_index]
                        )
                        ue_RL_list[ue_index].do_store_energy(
                            episode,
                            time_index,
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy [time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index]
                        )
                        reward_indicator[time_index, ue_index] = 1


            # ADD STEP (one step does not mean one store)
            RL_step += 1

            # UPDATE OBSERVATION
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # CONTROL LEARNING START TIME AND FREQUENCY
            if (RL_step > 200) and (RL_step % 10 == 0):
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn()

            # GAME ENDS
            if done:
                for task in env.task_history:
                    cmpl = drp = 0
                    for t in task:
                        d_states = t['d_state']
                        if any(d < 0 for d in d_states):
                            t['state'] = 'D'
                            drp += 1
                        elif all(d > 0 for d in d_states):
                            t['state'] = 'C'
                            cmpl += 1
                full_complete_task = 0
                full_drop_task = 0
                complete_task = 0
                drop_task = 0
                for history in env.task_history:
                    for task in history:
                        if task['state'] == 'C':
                            full_complete_task += 1
                        elif task['state'] == 'D':
                            full_drop_task += 1
                        for component_state in task['d_state']:
                            if component_state == 1:
                                complete_task += 1
                            elif component_state == -1:
                                drop_task += 1
                cnt = len(env.task_history) * len(env.task_history[0]) * env.n_component

                print("++++++++++++++++++++++")
                print("drrop_rate   : ", full_drop_task/(cnt/env.n_component))
                print("full_drrop   : ", full_drop_task)
                print("full_complate: ", full_complete_task)
                print("complete_task: ", complete_task)
                print("drop_task:     ", drop_task)
                print("++++++++++++++++++++++")

                if episode % 999 == 0 and episode != 0:
                    os.mkdir("models" + "/" + str(episode))
                    for ue in range(env.n_ue):
                        ue_RL_list[ue].saver.save(ue_RL_list[ue].sess, "models/" + str(episode) +'/'+ str(ue) + "_X_model" +'/model.ckpt', global_step=episode)

                avg_reward_list.append(-(monitor_reward(ue_RL_list, episode)))
                if episode % 10 == 0:
                    avg_reward_list_2.append(sum(avg_reward_list[episode-10:episode])/10)
                    avg_delay_list_in_episode.append(monitor_delay(ue_RL_list, episode))
                    avg_energy_list_in_episode.append(monitor_energy(ue_RL_list, episode))
                    total_drop = full_drop_task
                    num_task_drop_list_in_episode.append(total_drop)

                    # Plotting and saving figures
                    '''
                    fig, axs = plt.subplots(4, 1, figsize=(8, 16))
                    axs[0].plot(avg_reward_list, '-')
                    axs[0].set_ylabel('LSTM')
                    axs[1].plot(avg_delay_list_in_episode, '-')
                    axs[1].set_ylabel('r')
                    axs[2].plot(avg_energy_list_in_episode, '-')
                    axs[2].set_ylabel('r')
                    axs[3].plot(num_task_drop_list_in_episode, '-')
                    axs[3].set_ylabel('r')
                    plt.savefig('figures.png')
                    '''

                    # Writing data to files
                    '''
                    data = [avg_reward_list, avg_delay_list_in_episode, avg_energy_list_in_episode, num_task_drop_list_in_episode]
                    filenames = ['reward.txt', 'delay.txt', 'energy.txt', 'drop.txt']
                    for i in range(len(data)):
                        with open(filenames[i], 'w') as f:
                            f.write('\n'.join(str(x) for x in data[i]))
                    '''
                # Process energy
                ue_bit_processed = sum(sum(env.ue_bit_processed))
                ue_comp_energy = sum(sum(env.ue_comp_energy))

                # Transmission energy
                ue_bit_transmitted = sum(sum(env.ue_bit_transmitted))
                ue_tran_energy = sum(sum(env.ue_tran_energy))

                # edge energy
                edge_bit_processed = sum(sum(env.edge_bit_processed))
                edge_comp_energy = sum(sum(env.edge_comp_energy))
                ue_idle_energy = sum(sum(env.ue_idle_energy))

                # Print results
                print(int(ue_bit_processed), ue_comp_energy, "local")
                print(int(ue_bit_transmitted), ue_tran_energy, "trans")
                print(int(sum(edge_bit_processed)),sum(edge_comp_energy), sum(ue_idle_energy), "edge")
                print("_________________________________________________")

                break # Training Finished


if __name__ == "__main__":

    # GENERATE ENVIRONMENT
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    ue_RL_list = list()
    for ue in range(Config.N_UE):
        ue_RL_list.append(DuelingDoubleDeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                                    learning_rate       = Config.LEARNING_RATE,
                                                    reward_decay        = Config.REWARD_DDECAY,
                                                    e_greedy            = Config.E_GREEDY,
                                                    replace_target_iter = Config.N_NETWORK_UPDATE,  # each 200 steps, update target net
                                                    memory_size         = Config.MEMORY_SIZE,  # maximum of memory
                                                    ))

    # LOAD MODEL
    '''
    for ue in range(Config.N_UE):
        ue_RL_list[ue].Initialize(ue_RL_list[ue].sess, ue)
    '''

    # TRAIN THE SYSTEM
    train(ue_RL_list, Config.N_EPISODE)


