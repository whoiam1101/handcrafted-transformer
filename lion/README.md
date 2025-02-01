# 🦁 Lion Optimizer Implementation

This repository provides an implementation of the **Lion Optimizer** (EvoLved Sign Momentum), as described in the paper [*"Lion: A New Optimization Algorithm for Training Deep Neural Networks"*](https://arxiv.org/abs/2302.06675). Lion combines the advantages of adaptive and momentum-based optimizers, offering improved performance and efficiency for training deep neural networks.

## Algorithm Overview

Lion uses sign-based updates instead of traditional gradient-based updates, integrating momentum and exponentially moving averages (EMA) for more efficient and stable parameter updates.

The core update rule is as follows:

<p align="center">
    <img src="../assets/lion_algorithm.png" alt="Project Image" width="100%">
</p>

Where:
- $\beta_1$ and $\beta_2$ are momentum coefficients,
- $\lambda$ is a regularization parameter,
- $\eta$ is the learning rate.

## Features
- Sign-based updates for improved efficiency.
- Memory-efficient and suitable for large-scale machine learning tasks.
- PyTorch-compatible implementation.
