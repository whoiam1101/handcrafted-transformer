# Lion Optimizer Implementation

This repository contains an implementation of the **Lion optimizer** (EvoLved Sign Momentum) as described in the paper [*"Lion: A New Optimization Algorithm for Training Deep Neural Networks"*](https://arxiv.org/abs/2302.06675). Lion is a novel optimization algorithm that combines the benefits of both adaptive and momentum-based optimizers, offering improved performance and efficiency in training deep neural networks.

## Introduction
The Lion optimizer is a simple yet powerful optimization algorithm that uses sign-based updates instead of traditional gradient-based updates. It is designed to be memory-efficient and computationally lightweight, making it suitable for large-scale machine learning tasks.

This implementation provides a PyTorch-compatible version of the Lion optimizer, along with examples demonstrating its usage in training neural networks.

## Installation
To use the Lion optimizer, clone this repository and install the required dependencies.

```bash
git clone https://github.com/whoiam1101/lion-optimizer.git
cd lion-optimizer
pip install -r requirements.txt