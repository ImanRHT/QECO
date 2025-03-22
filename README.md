# QECO

## A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing
[![GitHub release (latest)](https://img.shields.io/github/v/release/ImanRht/QOCO)](https://github.com/ImanRht/QOCO/releases)
[![DOI](https://zenodo.org/badge/672957541.svg)](https://zenodo.org/doi/10.5281/zenodo.10134418)
![GitHub repo size](https://img.shields.io/github/repo-size/ImanRht/QOCO)
[![GitHub stars](https://img.shields.io/github/stars/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/stargazers) 
[![GitHub forks](https://img.shields.io/github/forks/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/network/members) 
[![GitHub issues](https://img.shields.io/github/issues/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/issues) 
[![GitHub license](https://img.shields.io/github/license/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/blob/master/LICENSE) 

This repository contains the Python code for reproducing the decentralized QECO (QoE-Oriented Computation Offloading) algorithm, designed for Mobile Edge Computing (MEC) systems.



## Citation
I. Rahmati, H. Shahmansouri, and A. Movaghar, "[QECO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing](https://arxiv.org/pdf/2311.02525.pdf)".

``` 
@article{rahmati2025qeco,
  title={QECO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing},
  author={Rahmati, Iman and Shah-Mansouri, Hamed and Movaghar, Ali},
  journal={arXiv preprint arXiv:2311.02525},
  url={https://arxiv.org/abs/2311.02525},
  year={2024}
}
```

## Overview



QECO is designed to balance and prioritize QoE factors based on individual mobile device requirements while considering the dynamic workloads at the edge nodes. The QECO algorithm captures the dynamics of the MEC environment by integrating the **Dueling Double Deep Q-Network (D3QN)** model with Long **Short-Term Memory (LSTM)** networks. This algorithm address the QoE maximization problem by efficiently utilizing resources from both MDs and ENs.

  
- **D3QN**: By integrating both double Q-learning and dueling network architectures, D3QN overcomes overestimation bias in action-value predictions and accurately identifies the relative importance of states and actions. This improves the model’s ability to make accurate predictions, providing a foundation for enhanced offloading strategies.

- **LSTM**: Incorporating LSTM networks allows the model to continuously estimate dynamic work- loads at edge servers. This is crucial for dealing with limited global information and adapting to the uncertain MEC environment with multiple MDs and ENs. By predicting the future workload of edge servers, MDs can effectively adjust their offloading strategies to achieve higher QoE.

<div align="center">
  <img src="/assets/D3QN.png" alt="D3QN architecture" title="D3QN architecture" style="width:100%;"/>
</div>


## Contents
- [main.py](main.py): The main code, including training and testing structures, implemented using [Tensorflow 1.x](https://www.tensorflow.org/install/pip).
- [MEC_Env.py](MEC_Env.py): Contains the code for the mobile edge computing environment.
- [D3QN.py](DDQN.py): The code for QECO netwok model, implemented using [Tensorflow 1.x](https://www.tensorflow.org/install/pip).
- [Config.py](Config.py): Configuration file for MEC entities and neural network setup.




## Quick Start

1. **Clone the repository**:

``` bash
   git clone https://github.com/ImanRHT/QECO.git
   cd QECO
```

2. **Configure the MEC environment** in [Config.py](Config.py).

3. Make sure you have the required packages listed in the [requirements.txt](requirements.txt) file installed to ensure the project functions correctly.
4. **Run the training script***:

``` bash
   python main.py
```



## Convergence

![Performance_Chart](/assets/Performance_Chart__.png "Performance_Charts")




## Future Directions


- Addressing **single-agent non-stationarity issues** by leveraging **multi-agent DRL**.
- **Accelerating the learning of optimal offloading policies** by taking advantage of **Federated Learning** techniques in the training process. This will allow MDs to collectively contribute to improving the offloading model and enable continuous learning when new MDs join the network.
- Addressing partially observable environment issues by designing a decentralized **Partially Observable Markov Decision Process (Dec-POMDP)**.
- Extending the **Task Models** by considering interdependencies among tasks. This can be achieved by incorporating a **Task Call Graph Representation**.
- Implementation of the D3QN algorithm using **PyTorch**, focusing on efficient **parallelization** and enhanced model stability.





## Contributing 

We welcome contributions! Here’s how you can get involved:

1. **Fork the repository**: Create your own copy of the project.
2. **Clone Your Fork**:

``` bash
  git clone https://github.com/<your-username>/<repo-name>.git  
  cd <repo-name>
```

3. **Create a new branch**: Name your branch to reflect the changes you're making.
   
``` bash
  git checkout -b feature/<add-future-direction-support>
```

4. **Commit your changes**: Write clear and concise commit messages.

``` bash
  git add * 
  git commit -a -m "<add-future-direction-support>"  
``` 

5. **Push your branch**:

``` bash
  git push origin feature/<add-future-direction-support>
```

6. **Open a pull request**: Navigate to the repository and submit your pull request. Provide a detailed description of your work.

For **bug reports** or **feature requests**, open a GitHub issue [here](https://github.com/ImanRht/QOCO/issues).

## About Authors

- [Iman Rahmati](https://scholar.google.com/citations?user=yHWKp6MAAAAJ&hl=en&oi=sra): Research Assistant in the Computer Science and Engineering Department at SUT.
- [Hamed Shah-Mansouri](https://scholar.google.com/citations?user=dcjIFccAAAAJ&hl=en&oi=ao): Assistant Professor in the Electrical Engineering Department at SUT.
- [Ali Movaghar](https://scholar.google.com/citations?user=BXNelwwAAAAJ&hl=en): Professor in the Computer Science and Engineering Department at SUT.



## Primary References

- H. Shah-Mansouri and V. W. Wong, “[Hierarchical fog-cloud computing for iot systems: A computation offloading game](https://ieeexplore.ieee.org/document/8360511)", IEEE Internet of Things Journal, May 2018.

- M. Tang and V. W. Wong, "[Deep reinforcement learning for task offloading in mobile edge computing systems](https://ieeexplore.ieee.org/abstract/document/9253665)", IEEE Transactions on Mobile Computing, Nov 2020.

- H. Zhou, K. Jiang, X. Liu, X. Li, and V. C. Leung, “[Deep reinforcement learning for energy-efficient computation offloading in mobile-edge computing](https://ieeexplore.ieee.org/document/9462445)”, IEEE Internet of Things Journal, Jun 2021.

- L. Yang, H. Zhang, X. Li, H. Ji, and V. C. Leung, “[A distributed computation offloading strategy in small-cell networks integrated with mobile edge computing](https://ieeexplore.ieee.org/document/8519737)”, IEEE/ACM Transactions on Networking, Dec 2018.

