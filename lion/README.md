# Lion Optimizer Implementation 🦁

This repository contains an implementation of the **Lion optimizer** (EvoLved Sign Momentum) as described in the paper [*"Lion: A New Optimization Algorithm for Training Deep Neural Networks"*](https://arxiv.org/abs/2302.06675). Lion is a novel optimization algorithm that combines the benefits of both adaptive and momentum-based optimizers, offering improved performance and efficiency in training deep neural networks.

## Introduction
The Lion optimizer is a simple yet powerful optimization algorithm that uses sign-based updates instead of traditional gradient-based updates. It is designed to be memory-efficient and computationally lightweight, making it suitable for large-scale machine learning tasks.

This implementation provides a PyTorch-compatible version of the Lion optimizer, along with examples demonstrating its usage in training neural networks.

## A Pseudocode for Lion
$$
\begin{aligned}
&\mathbf{given}\ \beta_1,\beta_2,\lambda,\eta,f\\
&\mathbf{initialize}\ \theta_0,m_0\leftarrow 0\\
&\mathbf{while}\ \theta_t\ \text{not converged}\ \mathbf{do}\\
&\qquad g_t\leftarrow \nabla_\theta f(\theta_{t-1})\\
&\qquad\mathbf{update\ model\ parameters}\\
&\qquad c_t\leftarrow \beta_1 m_{t-1}+(1-\beta_1)g_t\\
&\qquad \theta_t\leftarrow \theta_{t-1}-\eta(\mathrm{sign}(c_t)+\lambda \theta_{t-1})\\
&\qquad\mathbf{update\ EMA\ of}\  g_t\\
&\qquad m_t\leftarrow \beta_2 m_{t-1}+(1-\beta_2)g_t\\
&\mathbf{end\ while}\\
&\mathbf{return}\ \theta_t\\
\end{aligned}
$$
