"""
pinn_audio.physics
Utilities for nondimensionalization and physics loss using autograd derivatives.

Key functions:
- compute_dimensionless(p, P0)
- physics_residual(p_hat, tprime, alpha_prime, beta_prime)
- physics_loss(residual)
"""

import torch
import torch.nn.functional as F
from torch import nn


def safe_divide(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    return a / (b + eps)


def compute_dimensionless(p: torch.Tensor, p0: torch.Tensor):
    """
    p: [B,1,T]
    p0: [B,1,1]
    returns p_prime: [B,1,T]
    """
    return safe_divide(p, p0.unsqueeze(-1))


def physics_residual(p_hat: torch.Tensor, tprime: torch.Tensor, alpha_prime: torch.Tensor, beta_prime: torch.Tensor):
    """
    p_hat: dimensionless prediction [B,1,T] (p')
    tprime: [B,1,T] with requires_grad=True
    alpha_prime, beta_prime: scalars or broadcastable tensors (use softplus externally if learnable)
    returns residual [B,1,T] = p'' + alpha' p' + beta' p'
    """
    # p_hat: requires grad through network outputs
    # tprime requires_grad=True
    # compute first derivative d/dt' p_hat
    # torch.autograd.grad expects outputs dependent on tprime (it is) - use ones_like as grad_outputs
    grad_outputs = torch.ones_like(p_hat)
    # first derivative
    pt = torch.autograd.grad(p_hat, tprime, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
    # second derivative
    ptt = torch.autograd.grad(pt, tprime, grad_outputs=torch.ones_like(pt), create_graph=True)[0]
    # alpha_prime and beta_prime may be floats or tensors
    # ensure proper shapes for broadcasting
    res = ptt + alpha_prime * pt + beta_prime * p_hat
    return res


def physics_loss_from_residual(residual: torch.Tensor):
    return torch.mean(residual ** 2)
