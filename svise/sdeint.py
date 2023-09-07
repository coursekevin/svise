import torch
from torch import Tensor
import torchsde
from .sde_learning import SDELearner
from typing import Callable


class _SDE(torch.nn.Module):
    # noise_type = "general"
    noise_type = "additive"
    sde_type = "ito"

    def __init__(self, drift: Callable, diffusion: Callable):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    # Drift
    def f(self, t, y):
        # print(t)
        # shape (batch_size, state_size)
        return self.drift(t, y)

    # Diffusion
    def g(self, t, y):
        # shape (bastch_size, state_size, state_size)
        return self.diffusion()


def solve_sde(sde: SDELearner, x0: Tensor, t_eval: Tensor, **kwargs) -> Tensor:
    """Wraps torchsde.sdeint to solve the approximate posterior over the
    drift + diffusion function in the sde

    Args:
        sde (SDELearner): some SDE
        x0 (Tensor): intial conditions
        t_eval (Tensor): batch of time stamps
        **kwargs: kwargs passed onto the torchsde solver

    Returns:
        Tensor: sde solution samples
    """
    sde_tmp = _SDE(sde.drift, sde.diffusion)
    # perhaps we should check if |sde.diffusion| << 1 and then use a standard ode solver
    # note I have dug into the source code, this will work fine even for SDEs with random coefficients
    xs = torchsde.sdeint(sde_tmp, x0, t_eval, **kwargs)
    return xs
