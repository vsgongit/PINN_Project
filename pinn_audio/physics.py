"""
Physics utilities for temporal PINN audio denoiser.

Includes:
 - compute_dimensionless(p, p0, T0=None, sr=None, return_tprime_batched=False)
 - compute_dt_from_tprime(tprime)
 - finite_diff_1st / finite_diff_2nd
 - physics_residual(p_hat_prime, tprime, alpha_prime, beta_prime)
 - physics_loss_from_residual(residual)
"""
from typing import Tuple, Optional
import torch
import torch.nn.functional as F

EPS = 1e-8

def compute_dimensionless(p: torch.Tensor, p0: torch.Tensor, T0: Optional[float]=None, sr: Optional[int]=None, return_tprime_batched: bool=False):
    """
    Convert dimensional waveform to dimensionless form.

    Args:
      p: tensor [B, C, T] (waveform in physical units)
      p0: tensor [B, 1, 1] (scale; e.g. RMS per window). If zero-ish, p0 fallback will be applied.
      T0: window duration in seconds (optional). If provided together with sr, it is validated; otherwise t' is in [0,1] over samples.
      sr: sample rate (optional). If provided and T0 is None, T0 = T / sr is used.
      return_tprime_batched: if True, return tprime with shape [B,1,T], otherwise [1,1,T]

    Returns:
      p_prime: [B, C, T] dimensionless
      tprime: [1,1,T] or [B,1,T] dimensionless time grid
    """
    # ensure shapes
    if p.dim() == 2:
        p = p.unsqueeze(1)  # [B,1,T] assumption if B is first dim
    if p.dim() == 3 and p.shape[1] != 1:
        # keep channels as-is, but routine expects mono; still works
        pass

    B, C, T = p.shape
    # Normalize p0 shape
    if p0 is None:
        # fallback: compute RMS per-sample
        p0 = torch.sqrt(torch.clamp(torch.mean(p**2, dim=-1, keepdim=True), min=EPS))
    else:
        # ensure tensor and device
        if not torch.is_tensor(p0):
            p0 = torch.tensor(p0, dtype=p.dtype, device=p.device)
        p0 = p0.to(p.device).to(p.dtype)
        # broadcast to [B,1,1]
        if p0.dim() == 1:
            p0 = p0.view(-1,1,1)
        elif p0.dim() == 2:
            p0 = p0.unsqueeze(-1)

    # avoid division by tiny p0
    p0_safe = p0 + EPS

    p_prime = p / p0_safe

    # tprime: dimensionless time from 0..1
    t = torch.linspace(0.0, 1.0, steps=T, dtype=p.dtype, device=p.device)
    tprime = t.unsqueeze(0).unsqueeze(0)  # [1,1,T]
    if return_tprime_batched:
        tprime = tprime.expand(B, -1, -1).clone()

    return p_prime, tprime

def compute_dt_from_tprime(tprime: torch.Tensor) -> float:
    tp = tprime
    if tp.dim() == 4 and tp.shape[2] == 1:
        tp = tp.squeeze(2)
    if tp.dim() >= 3:
        tp1 = tp.reshape(-1, tp.shape[-1])
    else:
        tp1 = tp.unsqueeze(0)
    if tp1.shape[-1] < 2:
        raise ValueError("tprime must have at least 2 time samples to compute dt")
    diffs = tp1[:, 1:] - tp1[:, :-1]
    dt = diffs.mean().item()
    return dt

def finite_diff_1st(p: torch.Tensor, dt: float) -> torch.Tensor:
    T = p.shape[-1]
    pd = torch.zeros_like(p)
    if T >= 3:
        pd[..., 1:-1] = (p[..., 2:] - p[..., :-2]) / (2.0 * dt)
        pd[..., 0] = (p[..., 1] - p[..., 0]) / dt
        pd[..., -1] = (p[..., -1] - p[..., -2]) / dt
    elif T == 2:
        pd[..., 0] = (p[..., 1] - p[..., 0]) / dt
        pd[..., 1] = pd[..., 0].clone()
    else:
        pd = torch.zeros_like(p)
    return pd

def finite_diff_2nd(p: torch.Tensor, dt: float) -> torch.Tensor:
    T = p.shape[-1]
    p2 = torch.zeros_like(p)
    if T >= 3:
        p2[..., 1:-1] = (p[..., 2:] - 2.0 * p[..., 1:-1] + p[..., :-2]) / (dt * dt)
        p2[..., 0] = (p[..., 2] - 2.0 * p[..., 1] + p[..., 0]) / (dt * dt) if T > 2 else 0.0
        p2[..., -1] = (p[..., -1] - 2.0 * p[..., -2] + p[..., -3]) / (dt * dt) if T > 2 else 0.0
    elif T == 2:
        p2[..., 0] = p2[..., 1] = (p[..., 1] - p[..., 0]) / (dt * dt)
    else:
        p2 = torch.zeros_like(p)
    return p2

def physics_residual(p_hat_prime: torch.Tensor, tprime: torch.Tensor, alpha_prime, beta_prime) -> torch.Tensor:
    p = p_hat_prime
    if p.dim() != 3:
        p = p.reshape(p.shape[0], p.shape[1], -1)

    if tprime.dim() == 4 and tprime.shape[2] == 1:
        tprime_use = tprime.squeeze(2)
    else:
        tprime_use = tprime
    dt_prime = compute_dt_from_tprime(tprime_use)

    pt = finite_diff_1st(p, dt_prime)   # [B,1,T]
    ptt = finite_diff_2nd(p, dt_prime)  # [B,1,T]

    ap = alpha_prime
    bp = beta_prime
    if not torch.is_tensor(ap):
        ap = torch.tensor(float(ap), device=p.device, dtype=p.dtype)
    else:
        ap = ap.to(p.device).to(p.dtype)
    if not torch.is_tensor(bp):
        bp = torch.tensor(float(bp), device=p.device, dtype=p.dtype)
    else:
        bp = bp.to(p.device).to(p.dtype)

    residual = ptt + ap * pt + bp * p
    return residual

def physics_loss_from_residual(residual: torch.Tensor) -> torch.Tensor:
    return torch.mean(residual ** 2)
