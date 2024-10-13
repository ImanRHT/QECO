

class Config(object):
    N_UE        = 15
    N_EDGE      = 2
    N_COMPONENT = 1
    N_EPISODE   = 1000
    N_TIME_SLOT = 100     
    MAX_DELAY   = 10
    N_TIME      = N_TIME_SLOT + MAX_DELAY

    UE_COMP_ENERGY   = 2
    UE_TRAN_ENERGY   = 2.3
    UE_IDLE_ENERGY   = 0.1
    EDGE_COMP_ENERGY = 5

    DURATION         = 0.1
    UE_COMP_CAP      = 2.5
    UE_TRAN_CAP      = 14
    EDGE_COMP_CAP    = 41.8

    TASK_COMP_DENS   = 0.297
    TASK_ARRIVE_PROB = 0.3
    TASK_MIN_SIZE    = 2
    TASK_MAX_SIZE    = 5

    LEARNING_RATE    = 0.01
    REWARD_DDECAY    = 0.9
    E_GREEDY         = 0.99
    N_NETWORK_UPDATE = 200
    MEMORY_SIZE      = 500

