# QOCO

## A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing

[![GitHub release (latest)](https://img.shields.io/github/v/release/ImanRht/QOCO)](https://github.com/ImanRht/QOCO/releases)
![GitHub repo size](https://img.shields.io/github/repo-size/ImanRht/QOCO)
[![GitHub stars](https://img.shields.io/github/stars/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/stargazers) 
[![GitHub forks](https://img.shields.io/github/forks/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/network/members) 
[![GitHub issues](https://img.shields.io/github/issues/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/issues) 
[![GitHub license](https://img.shields.io/github/license/ImanRht/QOCO?style=social)](https://github.com/ImanRht/QOCO/blob/master/LICENSE) 

This repository contains the Python code for reproducing the decentralized QOCO (QoE-Oriented Computation Offloading) algorithm, designed for Mobile Edge Computing systems. QOCO leverages Deep Reinforcement Learning to empower mobile devices to make their offloading decisions and select offloading targets, with the aim of maximizing the long-term Quality of Experience (QoE) for each user individually.

## Contents

- [main.py](main.py): The main code, including training and testing structures, implemented using [Tensorflow 1.x](https://www.tensorflow.org/install/pip).
- [MEC_Env.py](MEC_Env.py): Contains the code for the mobile edge computing environment.
- [DDQN.py](DDQN.py): The code for reinforcement learning with double deep Q networks for mobile devices, implemented using [Tensorflow 1.x](https://www.tensorflow.org/install/pip).
- [DDQN_keras.py](DDQN_keras.py): Double deep Q network implementation using [Keras](https://keras.io/).
- [DDQN_torch.py](DDQN_torch.py): Double deep Q network implementation using [PyTorch](https://pytorch.org/get-started/locally/).
- [Config.py](Config.py): Configuration file for MEC entities and neural network setup.

## Cite this Work

If you use this work in your research, please cite it as follows:

I. Rahmati, H. Shahmansouri, and A. Movaghar, "[QOCO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing](https://arxiv.org/pdf/2311.02525.pdf)", submitted to IEEE Internet of Things Journal, Oct 2023.

```
@article{rahmati2023qoco,
  title={QOCO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing},
  author={Rahmati, Iman and Shah-Mansouri, Hamed and Movaghar, Ali},
  journal={arXiv preprint arXiv:2311.02525},
  year={2023}
}
```

## About Authors

- [Iman Rahmati](): Research Assistant in the Computer Science and Engineering Department at SUT.
- [Hamed Shah-Mansouri](https://scholar.google.com/citations?user=dcjIFccAAAAJ&hl=en&oi=ao): Assistant Professor in the Electrical Engineering Department at SUT.
- [Ali Movaghar](https://scholar.google.com/citations?user=BXNelwwAAAAJ&hl=en): Professor in the Computer Science and Engineering Department at SUT.

## Required Packages

Make sure you have the following packages installed:

- [Tensorflow](https://www.tensorflow.org/install/pip)
- [PyTorch](https://pytorch.org/get-started/locally/)
- numpy
- matplotlib


## Contribute
If you have an issue or found a bug, please raise a GitHub issue [here](https://github.com/ImanRht/QOCO/issues). Pull requests are also welcome.

