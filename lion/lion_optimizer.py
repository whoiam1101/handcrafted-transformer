import torch

from torch import Tensor
from torch.optim.optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _fused_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _stack_if_compiling,
    _use_grad_for_differentiable,
    _view_as_real,
    DeviceDict,
    DeviceDtypeDict,
    Optimizer,
    ParamsT,
)


__all__ = ["Lion", "lion"]


class Lion(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 0.001,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None
    ):
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (
            (isinstance(betas[0], float) and isinstance(betas[1], float))
            or (isinstance(betas[0], Tensor) and isinstance(betas[1], Tensor))
        ):
            raise ValueError("betas must be either both floats or both Tensors")
        if isinstance(betas[0], Tensor):
            if not capturable and foreach:
                raise ValueError(
                    "betas[0] as a Tensor is not supported for capturable=False and foreach=True"
                )
            if betas[0].numel() != 1:
                raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], Tensor):
            if not capturable and foreach:
                raise ValueError(
                    "betas[1] as a Tensor is not supported for capturable=False and foreach=True"
                )
            if betas[1].numel() != 1:
                raise ValueError("Tensor betas[1] must be 1-element")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )
