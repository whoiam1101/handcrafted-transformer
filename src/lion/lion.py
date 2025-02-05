import torch

from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from typing import Callable


class Lion(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-4,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> float | None:
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            decay_factor = 1.0 - lr * weight_decay

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                exp_avgs.append(state["exp_avg"])

            if not params_with_grad:
                continue

            params_data = [p.data for p in params_with_grad]

            update_part1 = torch._foreach_mul(exp_avgs, beta1)
            update_part2 = torch._foreach_mul(grads, 1 - beta1)
            updates = torch._foreach_add(update_part1, update_part2)

            if decay_factor != 1.0:
                torch._foreach_mul_(params_data, decay_factor)

            sign_updates = [u.sign() for u in updates]
            torch._foreach_add_(params_data, sign_updates, alpha=-lr)

            torch._foreach_mul_(exp_avgs, beta2)
            grad_part = torch._foreach_mul(grads, 1 - beta2)
            torch._foreach_add_(exp_avgs, grad_part)

        return loss


Lion.__doc__ = """
Lion optimizer for training neural networks.

Args:
    params (ParamsT): Parameters to optimize.
    lr (float | Tensor, optional): Learning rate. Default is 1e-4.
    betas (tuple[float | Tensor, float | Tensor], optional): Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.999).
    weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.

Methods:
    step(closure: callable | None = None) -> float | None:
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss. Default is None.
        Returns:
            float | None: The loss value if closure is provided, otherwise None.
"""
