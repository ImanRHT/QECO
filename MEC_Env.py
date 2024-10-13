from Config import Config
import numpy as np
import random
import math
import queue

class MEC:
    def __init__(self, num_ue, num_edge, num_time, num_component, max_delay):
        # Initialize variables
        self.n_ue          = num_ue
        self.n_edge        = num_edge
        self.n_time        = num_time
        self.n_component   = num_component
        self.max_delay     = max_delay
        self.duration      = Config.DURATION
        self.ue_p_comp     = Config.UE_COMP_ENERGY
        self.ue_p_tran     = Config.UE_TRAN_ENERGY
        self.ue_p_idle     = Config.UE_IDLE_ENERGY
        self.edge_p_comp   = Config.EDGE_COMP_ENERGY

        self.time_count      = 0
        self.task_count_ue   = 0
        self.task_count_edge = 0
        self.n_actions       = self.n_component + 1
        self.n_features      = 1 + 1 + 1 + self.n_edge
        self.n_lstm_state    = self.n_edge
  
        # Computation and transmission capacities
        self.comp_cap_ue   = Config.UE_COMP_CAP * np.ones(self.n_ue) * self.duration
        self.comp_cap_edge = Config.EDGE_COMP_CAP * np.ones([self.n_edge]) * self.duration
        self.tran_cap_ue   = Config.UE_TRAN_CAP * np.ones([self.n_ue, self.n_edge]) * self.duration
        

        #self.comp_density  = [Config.TASK_COMP_DENS[np.random.randint(0, len(Config.TASK_COMP_DENS))] for _ in range(self.n_ue)]

        self.comp_density  = Config.TASK_COMP_DENS[1] * np.ones([self.n_ue])

        self.n_cycle = 1
        self.task_arrive_prob = Config.TASK_ARRIVE_PROB
        self.max_arrive_size   = Config.TASK_MAX_SIZE
        self.min_arrive_size   = Config.TASK_MIN_SIZE
        self.arrive_task_size_set    = np.arange(self.min_arrive_size, self.max_arrive_size, 0.1)

        
        self.arrive_task_size   = np.zeros([self.n_time, self.n_ue])

        self.arrive_task_dens   = np.zeros([self.n_time, self.n_ue])



        self.n_task = int(self.n_time * self.task_arrive_prob)

        # Task delay and energy-related arrays
        self.process_delay = np.zeros([self.n_time, self.n_ue])
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])

        # Queue information initialization
        self.t_ue_comp = -np.ones([self.n_ue])
        self.t_ue_tran = -np.ones([self.n_ue])
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])

        # Queue initialization
        self.ue_computation_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.ue_transmission_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.edge_computation_queue = [[queue.Queue() for _ in range(self.n_edge)] for _ in range(self.n_ue)]
        self.edge_ue_m = np.zeros(self.n_edge)
        self.edge_ue_m_observe = np.zeros(self.n_edge)

        # Task indicator initialization
        self.local_process_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in range(self.n_ue)]
        self.local_transmit_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in range(self.n_ue)]
        self.edge_process_task = [[{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'REMAIN': np.nan} for _ in range(self.n_edge)] for _ in range(self.n_ue)]

        self.task_history = [[] for _ in range(self.n_ue)]

    def reset(self, arrive_task_size):
        # Reset variables and queues
        self.task_history = [[] for _ in range(self.n_ue)]
        self.UE_TASK = [-1] * self.n_ue
        self.drop_edge_count = 0
        self.arrive_task_size = arrive_task_size
        self.time_count = 0
        self.local_process_task = []
        self.local_transmit_task = []
        self.edge_process_task = []
        self.t_ue_comp = -np.ones([self.n_ue])
        self.t_ue_tran = -np.ones([self.n_ue])
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])
        self.ue_computation_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.ue_transmission_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.edge_computation_queue = [[queue.Queue() for _ in range(self.n_edge)] for _ in range(self.n_ue)]
        self.process_delay = np.zeros([self.n_time, self.n_ue])
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])
        self.local_process_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in range(self.n_ue)]
        self.local_transmit_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan} for _ in range(self.n_ue)]
        self.edge_process_task = [[{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'REMAIN': np.nan} for _ in range(self.n_edge)] for _ in range(self.n_ue)]

        # Initial observation and LSTM state
        UEs_OBS = np.zeros([self.n_ue, self.n_features])
        for ue_index in range(self.n_ue):
            if self.arrive_task_size[self.time_count, ue_index] != 0:
                UEs_OBS[ue_index, :] = np.hstack([
                    self.arrive_task_size[self.time_count, ue_index], self.t_ue_comp[ue_index],
                    self.t_ue_tran[ue_index],
                    np.squeeze(self.b_edge_comp[ue_index, :])])

        UEs_lstm_state = np.zeros([self.n_ue, self.n_lstm_state])

        return UEs_OBS, UEs_lstm_state

    # perform action, observe state and delay (several steps later)
    def step(self, action):
    
        # EXTRACT ACTION FOR EACH ue
        ue_action_local = np.zeros([self.n_ue], np.int32)
        ue_action_offload = np.zeros([self.n_ue], np.int32)
        ue_action_edge = np.zeros([self.n_ue], np.int32)
        ue_action_component = np.zeros([self.n_ue], np.int32)-1

        random_list  = []
        for i in range(self.n_component):
            random_list.append(i)

        # UE QUEUES UPDATE
        for ue_index in range(self.n_ue):
            component_list = np.zeros([self.n_component], np.int32) - 1
            state_list = np.zeros([self.n_component], np.int32)
            ue_action = action[ue_index]
            
            if ue_action == 0:
                ue_action_local[ue_index] = 1
            else:
                ue_action_offload[ue_index] = 1
                sample = random.sample(random_list, int(ue_action))
                for i in range(len(sample)):
                    component_list[sample[i]] = np.random.randint(0, self.n_edge)
            
            ue_action_component[ue_index] = action[ue_index]
            ue_comp_cap = np.squeeze(self.comp_cap_ue[ue_index])
            ue_comp_density = self.comp_density[ue_index]
            ue_tran_cap = np.squeeze(self.tran_cap_ue[ue_index, :])[1] / self.n_cycle
            ue_arrive_task_size = np.squeeze(self.arrive_task_size[self.time_count, ue_index])
            
            if ue_arrive_task_size > 0:
                self.UE_TASK[ue_index] += 1
                task_dic = {
                    'UE_ID': ue_index,
                    'TASK_ID': self.UE_TASK[ue_index],
                    'SIZE': ue_arrive_task_size,
                    'TIME': self.time_count,
                    'EDGE': component_list,
                    'd_state': state_list,
                    'state': np.nan
                }
                self.task_history[ue_index].append(task_dic)
            
            for component in range(self.n_component):
                temp_dic = {
                    'DIV': component,
                    'UE_ID': ue_index,
                    'TASK_ID': self.UE_TASK[ue_index],
                    'SIZE': ue_arrive_task_size / self.n_component,
                    'TIME': self.time_count,
                    'EDGE': component_list[component],
                    'd_state': state_list[component]
                }
                
                if component_list[component] > -1:
                    self.ue_transmission_queue[ue_index].put(temp_dic)
                else:
                    self.ue_computation_queue[ue_index].put(temp_dic)
            
            for cycle in range(self.n_cycle):
                ue_comp_cap = np.squeeze(self.comp_cap_ue[ue_index]) / self.n_cycle

                if ((math.isnan(self.local_process_task[ue_index]['REMAIN']) and (not self.ue_computation_queue[ue_index].empty())) or
                    (math.isnan(self.local_transmit_task[ue_index]['REMAIN']) and (not self.ue_transmission_queue[ue_index].empty()))):

                    # Process UE computation queue
                    if not self.ue_computation_queue[ue_index].empty():
                        while not self.ue_computation_queue[ue_index].empty():
                            task = self.ue_computation_queue[ue_index].get()
                            if task['SIZE'] != 0:
                                if self.time_count - task['TIME'] + 1 <= self.max_delay:
                                    self.local_process_task[ue_index].update({
                                        'UE_ID': task['UE_ID'],
                                        'TASK_ID': task['TASK_ID'],
                                        'SIZE': task['SIZE'],
                                        'TIME': task['TIME'],
                                        'REMAIN': task['SIZE'],
                                        'DIV': task['DIV'],
                                    })
                                    break
                                else:
                                    self.task_history[ue_index][task['TASK_ID']]['d_state'][task['DIV']] = -1
                                    self.process_delay[task['TIME'], ue_index] = self.max_delay
                                    self.unfinish_task[task['TIME'], ue_index] = 1
                    
                    # Process UE transmission queue
                    if not self.ue_transmission_queue[ue_index].empty():
                        while not self.ue_transmission_queue[ue_index].empty():
                            task = self.ue_transmission_queue[ue_index].get()
                            if task['SIZE'] != 0:
                                if self.time_count - task['TIME'] + 1 <= self.max_delay:
                                    self.local_transmit_task[ue_index].update({
                                        'UE_ID': task['UE_ID'],
                                        'TASK_ID': task['TASK_ID'],
                                        'SIZE': task['SIZE'],
                                        'TIME': task['TIME'],
                                        'EDGE': int(task['EDGE']),
                                        'REMAIN': self.local_transmit_task[ue_index]['SIZE'],
                                        'DIV': task['DIV'],
                                    })
                                    break
                                else:
                                    self.task_history[task['UE_ID']][task['TASK_ID']]['d_state'][task['DIV']] = -1
                                    self.process_delay[task['TIME'], ue_index] = self.max_delay
                                    self.unfinish_task[task['TIME'], ue_index] = 1

                # PROCESS
                if self.local_process_task[ue_index]['REMAIN'] > 0:
                    if self.local_process_task[ue_index]['REMAIN'] >= (ue_comp_cap / ue_comp_density):
                        self.ue_bit_processed[self.local_process_task[ue_index]['TIME'], ue_index] += ue_comp_cap / ue_comp_density
                        self.ue_comp_energy[self.local_process_task[ue_index]['TIME'], ue_index] += (
                            (ue_comp_cap / ue_comp_density) * self.ue_p_comp * ue_comp_density
                        ) / (self.comp_cap_ue[ue_index])
                    else:
                        self.ue_bit_processed[self.local_process_task[ue_index]['TIME'], ue_index] += self.local_process_task[ue_index]['REMAIN']
                        self.ue_comp_energy[self.local_process_task[ue_index]['TIME'], ue_index] += (
                            self.local_process_task[ue_index]['REMAIN'] * self.ue_p_comp * ue_comp_density
                        ) / (self.comp_cap_ue[ue_index])

                    self.local_process_task[ue_index]['REMAIN'] = self.local_process_task[ue_index]['REMAIN'] - (ue_comp_cap / ue_comp_density)
                
                    # if no remain, compute processing delay
                    if self.local_process_task[ue_index]['REMAIN'] <= 0:
                        self.task_history[ue_index][self.local_process_task[ue_index]['TASK_ID']]['d_state'][self.local_process_task[ue_index]['DIV']] = 1
                        self.local_process_task[ue_index]['REMAIN'] = np.nan
                        if sum(self.task_history[ue_index][self.local_process_task[ue_index]['TASK_ID']]['d_state']) > self.n_component - 1:
                            self.process_delay[self.local_process_task[ue_index]['TIME'], ue_index] = self.time_count - self.local_process_task[ue_index]['TIME'] + 1
                    elif self.time_count - self.local_process_task[ue_index]['TIME'] + 1 == self.max_delay:
                        self.task_history[ue_index][self.local_process_task[ue_index]['TASK_ID']]['d_state'][self.local_process_task[ue_index]['DIV']] = -1
                        self.local_process_task[ue_index]['REMAIN'] = np.nan
                        self.process_delay[self.local_process_task[ue_index]['TIME'], ue_index] = self.max_delay
                        self.unfinish_task[self.local_process_task[ue_index]['TIME'], ue_index] = 1
                
                if self.local_transmit_task[ue_index]['REMAIN'] > 0:
                    if self.local_transmit_task[ue_index]['REMAIN'] >= ue_tran_cap:
                        self.ue_bit_transmitted[self.local_transmit_task[ue_index]['TIME'], ue_index] += ue_tran_cap
                        self.ue_tran_energy[self.local_transmit_task[ue_index]['TIME'], ue_index] += (self.local_transmit_task[ue_index]['REMAIN'] * self.ue_p_tran) / self.tran_cap_ue[0][0]
                    else:
                        self.ue_bit_transmitted[self.local_transmit_task[ue_index]['TIME'], ue_index] += self.local_transmit_task[ue_index]['REMAIN']
                        self.ue_tran_energy[self.local_transmit_task[ue_index]['TIME'], ue_index] += (self.local_transmit_task[ue_index]['REMAIN'] * self.ue_p_tran) / self.tran_cap_ue[0][0]
                    self.local_transmit_task[ue_index]['REMAIN'] = self.local_transmit_task[ue_index]['REMAIN'] - ue_tran_cap 

                    # UPDATE edge QUEUE
                    if self.local_transmit_task[ue_index]['REMAIN'] <= 0:
                        tmp_dict = {
                            'UE_ID': self.local_transmit_task[ue_index]['UE_ID'],
                            'TASK_ID': self.local_transmit_task[ue_index]['TASK_ID'],
                            'SIZE': self.local_transmit_task[ue_index]['SIZE'],
                            'TIME': self.local_transmit_task[ue_index]['TIME'],
                            'EDGE': self.local_transmit_task[ue_index]['EDGE'],
                            'DIV': self.local_transmit_task[ue_index]['DIV']
                        }
                        self.edge_computation_queue[ue_index][self.local_transmit_task[ue_index]['EDGE']].put(tmp_dict)
                        self.task_count_edge = self.task_count_edge + 1
                        edge_index = self.local_transmit_task[ue_index]['EDGE']
                        self.b_edge_comp[ue_index, edge_index] = self.b_edge_comp[ue_index, edge_index] + self.local_transmit_task[ue_index]['SIZE']
                        self.process_delay_trans[self.local_transmit_task[ue_index]['TIME'], ue_index] = self.time_count - self.local_transmit_task[ue_index]['TIME'] + 1
                        self.local_transmit_task[ue_index]['REMAIN'] = np.nan
                    elif self.time_count - self.local_transmit_task[ue_index]['TIME'] + 1 == self.max_delay:
                        self.task_history[self.local_transmit_task[ue_index]['UE_ID']][self.local_transmit_task[ue_index]['TASK_ID']]['d_state'][self.local_transmit_task[ue_index]['DIV']] = -1
                        self.local_transmit_task[ue_index]['REMAIN'] = np.nan
                        self.process_delay[self.local_transmit_task[ue_index]['TIME'], ue_index] = self.max_delay
                        self.unfinish_task[self.local_transmit_task[ue_index]['TIME'], ue_index] = 1
            
            if ue_arrive_task_size != 0:
                tmp_tilde_t_ue_comp = max(self.t_ue_comp[ue_index] + 1, self.time_count)
                comp_time = math.ceil(ue_arrive_task_size * ue_action_local[ue_index] / (np.squeeze(self.comp_cap_ue[ue_index]) / ue_comp_density))
                self.t_ue_comp[ue_index] = min(tmp_tilde_t_ue_comp + comp_time - 1, self.time_count + self.max_delay - 1)
                tmp_tilde_t_ue_tran = max(self.t_ue_tran[ue_index] + 1, self.time_count)
                tran_time = math.ceil(ue_arrive_task_size * (1 - ue_action_local[ue_index]) / np.squeeze(self.tran_cap_ue[ue_index,:])[1])
                self.t_ue_tran[ue_index] = min(tmp_tilde_t_ue_tran + tran_time - 1, self.time_count + self.max_delay - 1)

        # EDGE QUEUES UPDATE
        for ue_index in range(self.n_ue):
            ue_comp_density = self.comp_density[ue_index]
            for edge_index in range(self.n_edge):
                edge_cap = self.comp_cap_edge[edge_index] / self.n_cycle
                for cycle in range(self.n_cycle): 
                    # TASK ON PROCESS
                    if math.isnan(self.edge_process_task[ue_index][edge_index]['REMAIN']) and (not self.edge_computation_queue[ue_index][edge_index].empty()):
                        while not self.edge_computation_queue[ue_index][edge_index].empty():
                            task = self.edge_computation_queue[ue_index][edge_index].get()
                            if self.time_count - task['TIME'] + 1 <= self.max_delay:
                                self.edge_process_task[ue_index][edge_index].update({
                                    'UE_ID': task['UE_ID'],
                                    'TASK_ID': task['TASK_ID'],
                                    'SIZE': task['SIZE'],
                                    'TIME': task['TIME'],
                                    'REMAIN': self.edge_process_task[ue_index][edge_index]['SIZE'],
                                    'DIV': task['DIV'],
                                })
                                break
                            else:
                                self.task_history[task['UE_ID']][task['TASK_ID']]['d_state'][task['DIV']] = -1
                                self.process_delay[task['TIME'], ue_index] = self.max_delay
                                self.unfinish_task[task['TIME'], ue_index] = 1

                    # PROCESS
                    self.edge_drop[ue_index, edge_index] = 0
                    remaining_task = self.edge_process_task[ue_index][edge_index]['REMAIN']
                    if remaining_task > 0:
                        processed_amount = min(remaining_task, edge_cap / ue_comp_density / self.edge_ue_m[edge_index])
                        self.edge_bit_processed[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index, edge_index] += processed_amount
                        comp_energy = (self.edge_p_comp * processed_amount * ue_comp_density * pow(10, 9)) / (edge_cap * 10 * pow(10, 9))
                        idle_energy = (self.ue_p_idle * processed_amount * ue_comp_density * pow(10, 9)) / (edge_cap * 10 * pow(10, 9) / self.edge_ue_m[edge_index])
                        self.edge_comp_energy[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index, edge_index] += comp_energy
                        self.ue_idle_energy[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index, edge_index] += idle_energy
                        self.edge_process_task[ue_index][edge_index]['REMAIN'] -= processed_amount
                        # if no remain, compute processing delay
                        if self.edge_process_task[ue_index][edge_index]['REMAIN'] <= 0:
                            task_id = self.edge_process_task[ue_index][edge_index]['TASK_ID']
                            task_history = self.task_history[self.edge_process_task[ue_index][edge_index]['UE_ID']][task_id]
                            task_history['d_state'][self.edge_process_task[ue_index][edge_index]['DIV']] = 1
                            self.edge_process_task[ue_index][edge_index]['REMAIN'] = np.nan
                            if sum(task_history['d_state']) > self.n_component - 1:
                                self.process_delay[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index] = self.time_count - self.edge_process_task[ue_index][edge_index]['TIME'] + 1
                        elif self.time_count - self.edge_process_task[ue_index][edge_index]['TIME'] + 1 == self.max_delay:
                            task_id = self.edge_process_task[ue_index][edge_index]['TASK_ID']
                            task_history = self.task_history[self.edge_process_task[ue_index][edge_index]['UE_ID']][task_id]
                            task_history['d_state'][self.edge_process_task[ue_index][edge_index]['DIV']] = -1
                            self.edge_process_task[ue_index][edge_index]['REMAIN'] = np.nan
                            self.edge_drop[ue_index, edge_index] = remaining_task
                            self.process_delay[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index] = self.max_delay
                            self.unfinish_task[self.edge_process_task[ue_index][edge_index]['TIME'], ue_index] = 1

                # OTHER INFO
                if self.edge_ue_m[edge_index] != 0:
                    b_edge_comp_value = self.b_edge_comp[ue_index, edge_index]
                    b_edge_comp_value -= (edge_cap / ue_comp_density / self.edge_ue_m[edge_index] + self.edge_drop[ue_index, edge_index])
                    self.b_edge_comp[ue_index, edge_index] = max(b_edge_comp_value, 0)

        # COMPUTE CONGESTION (FOR NEXT TIME SLOT)
        self.edge_ue_m_observe = self.edge_ue_m
        self.edge_ue_m = np.zeros(self.n_edge)
        for edge_index in range(self.n_edge):
            for ue_index in range(self.n_ue):
                if (not self.edge_computation_queue[ue_index][edge_index].empty()) \
                        or self.edge_process_task[ue_index][edge_index]['REMAIN'] > 0:
                    self.edge_ue_m[edge_index] += 1

        # TIME UPDATE
        self.time_count = self.time_count + 1
        done = False
        if self.time_count >= self.n_time:
            done = True
            # set all the tasks' processing delay and unfinished indicator
            for time_index in range(self.n_time):
                for ue_index in range(self.n_ue):
                    if self.process_delay[time_index, ue_index] == 0 and self.arrive_task_size[time_index, ue_index] != 0:
                        self.process_delay[time_index, ue_index] = (self.time_count - 1) - time_index + 1
                        self.unfinish_task[time_index, ue_index] = 1
 
        # OBSERVATION
        UEs_OBS_ = np.zeros([self.n_ue, self.n_features])
        UEs_lstm_state_ = np.zeros([self.n_ue, self.n_lstm_state])
        if not done:
            for ue_index in range(self.n_ue):
                # observation is zero if there is no task arrival
                if self.arrive_task_size[self.time_count, ue_index] != 0:
                    # state [A, B^{comp}, B^{tran}, [B^{edge}]]
                    UEs_OBS_[ue_index, :] = np.hstack([
                        self.arrive_task_size[self.time_count, ue_index],
                        self.t_ue_comp[ue_index] - self.time_count + 1,
                        self.t_ue_tran[ue_index] - self.time_count + 1,
                        self.b_edge_comp[ue_index, :]])

                UEs_lstm_state_[ue_index, :] = np.hstack(self.edge_ue_m_observe)

        return UEs_OBS_, UEs_lstm_state_, done