# QECO

## A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing
[![GitHub release (latest)](https://img.shields.io/github/v/release/ImanRht/QOCO)](https://github.com/ImanRht/QOCO/releases)
[![DOI](https://zenodo.org/badge/672957541.svg)](https://zenodo.org/doi/10.5281/zenodo.10134418)
![GitHub repo size](https://img.shields.io/github/repo-size/ImanRht/QOCO)
[![GitHub stars](https://img.shields.io/github/stars/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/stargazers) 
[![GitHub forks](https://img.shields.io/github/forks/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/network/members) 
[![GitHub issues](https://img.shields.io/github/issues/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/issues) 
[![GitHub license](https://img.shields.io/github/license/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/blob/master/LICENSE) 

This repository contains the Python code for reproducing the decentralized QECO (QoE-Oriented Computation Offloading) algorithm, designed for Mobile Edge Computing systems. QECO leverages Deep Reinforcement Learning to empower mobile devices to make their offloading decisions and select offloading targets, with the aim of maximizing the long-term Quality of Experience (QoE) for each user individually.
## Contents
- [main.py](main.py): The main code, including training and testing structures, implemented using [Tensorflow 1.x](https://www.tensorflow.org/install/pip).
- [MEC_Env.py](MEC_Env.py): Contains the code for the mobile edge computing environment.
- [D3QN.py](DDQN.py): The code for reinforcement learning with double deep Q-network (D3QN) for mobile devices, implemented using [Tensorflow 1.x](https://www.tensorflow.org/install/pip).
- [DDQN_keras.py](DDQN_keras.py): D3QN implementation using [Keras](https://keras.io/).
- [DDQN_torch.py](DDQN_torch.py): D3QN implementation using [PyTorch](https://pytorch.org/get-started/locally/).
- [Config.py](Config.py): Configuration file for MEC entities and neural network setup.

## Cite this Work
If you use this work in your research, please cite it as follows:

I. Rahmati, H. Shahmansouri, and A. Movaghar, "[QECO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing](https://arxiv.org/pdf/2311.02525.pdf)".

```
@article{rahmati2023qeco,
  title={QECO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing},
  author={Rahmati, Iman and Shah-Mansouri, Hamed and Movaghar, Ali},
  journal={arXiv preprint arXiv:2311.02525},
  year={2023}
}
```

## About Authors

- [Iman Rahmati](https://scholar.google.com/citations?user=yHWKp6MAAAAJ&hl=en&oi=sra): Research Assistant in the Computer Science and Engineering Department at SUT.
- [Hamed Shah-Mansouri](https://scholar.google.com/citations?user=dcjIFccAAAAJ&hl=en&oi=ao): Assistant Professor in the Electrical Engineering Department at SUT.
- [Ali Movaghar](https://scholar.google.com/citations?user=BXNelwwAAAAJ&hl=en): Professor in the Computer Science and Engineering Department at SUT.

## Required Packages

Make sure you have the following packages installed:

- [Tensorflow](https://www.tensorflow.org/install/pip)
- [PyTorch](https://pytorch.org/get-started/locally/)
- numpy
- matplotlib


## Primary References

- H. Shah-Mansouri and V. W. Wong, “[Hierarchical fog-cloud computing for iot systems: A computation offloading game](https://ieeexplore.ieee.org/document/8360511)", IEEE Internet of Things Journal, May 2018.

- M. Tang and V. W. Wong, "[Deep reinforcement learning for task offloading in mobile edge computing systems](https://ieeexplore.ieee.org/abstract/document/9253665)", IEEE Transactions on Mobile Computing, Nov 2020.

- H. Zhou, K. Jiang, X. Liu, X. Li, and V. C. Leung, “[Deep reinforcement learning for energy-efficient computation offloading in mobile-edge computing](https://ieeexplore.ieee.org/document/9462445)”, IEEE Internet of Things Journal, Jun 2021.

- L. Yang, H. Zhang, X. Li, H. Ji, and V. C. Leung, “[A distributed computation offloading strategy in small-cell networks integrated with mobile edge computing](https://ieeexplore.ieee.org/document/8519737)”, IEEE/ACM Transactions on Networking, Dec 2018.


## Contribute
If you have an issue or found a bug, please raise a GitHub issue [here](https://github.com/ImanRht/QOCO/issues). Pull requests are also welcome.

