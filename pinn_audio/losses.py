"""
pinn_audio.losses
Data (MSE), SI-SDR, STFT mag loss utilities
"""

import torch
import torch.nn.functional as F
import math
import torchaudio


def mse_loss(pred: torch.Tensor, target: torch.Tensor):
    return F.mse_loss(pred, target)


def si_sdr(pred, target, eps=1e-8):
    """
    Scale-Invariant SDR (batch-wise)
    pred, target: [B,1,T]
    returns mean SI-SDR (in dB) across batch
    """

    def _reshape(x):
        return x.view(x.shape[0], -1)

    x = _reshape(target)
    s = _reshape(pred)
    x_mean = x.mean(dim=1, keepdim=True)
    s_mean = s.mean(dim=1, keepdim=True)
    x_z = x - x_mean
    s_z = s - s_mean
    pair_wise_dot = torch.sum(s_z * x_z, dim=1, keepdim=True)
    target_energy = torch.sum(x_z ** 2, dim=1, keepdim=True) + eps
    # projection
    scaled_target = pair_wise_dot * x_z / target_energy
    noise = s_z - scaled_target
    ratio = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(noise ** 2, dim=1) + eps)
    si_sdr_val = 10 * torch.log10(ratio + eps)
    return si_sdr_val.mean()


def stft_mag_loss(pred, target, n_fft=512, hop_length=128, win_length=512):
    """
    L2 loss between log-magnitude STFTs
    """
    window = torch.hann_window(win_length).to(pred.device)
    P = torch.stft(pred.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=window, return_complex=True, normalized=False)
    Q = torch.stft(target.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=window, return_complex=True, normalized=False)
    mag_p = torch.abs(P)
    mag_t = torch.abs(Q)
    # log-magnitude
    lm_p = torch.log1p(mag_p)
    lm_t = torch.log1p(mag_t)
    return F.mse_loss(lm_p, lm_t)
