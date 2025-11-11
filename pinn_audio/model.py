"""
pinn_audio.model
UNet1D-like encoder-decoder for waveform denoising.

Input: [B,1,T] waveform -> Output: [B,1,T] denoised waveform
"""

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=15, stride=1, padding=None, norm="g"):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.act = nn.SiLU()
        if norm == "g":
            self.norm = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        elif norm == "b":
            self.norm = nn.BatchNorm1d(out_ch)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UNet1D(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), kernel_size=15, use_skip=True):
        super().__init__()
        self.widths = widths
        self.use_skip = use_skip
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 1
        for w in widths:
            self.encs.append(ConvBlock(in_ch, w, kernel_size=kernel_size))
            self.pools.append(nn.Conv1d(w, w, kernel_size=4, stride=2, padding=1))  # downsample by 2
            in_ch = w
        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(widths[-1], widths[-1], kernel_size=kernel_size),
            ConvBlock(widths[-1], widths[-1], kernel_size=kernel_size),
        )
        # decoder
        self.upconvs = nn.ModuleList()
        self.decs = nn.ModuleList()
        rev = list(reversed(widths))
        for i in range(len(rev) - 1):
            self.upconvs.append(nn.ConvTranspose1d(rev[i], rev[i+1], kernel_size=4, stride=2, padding=1))
            self.decs.append(ConvBlock(rev[i] + rev[i+1] if use_skip else rev[i+1], rev[i+1], kernel_size=kernel_size))
        # final conv to 1 channel
        self.out_conv = nn.Conv1d(widths[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        cur = x
        for enc, pool in zip(self.encs, self.pools):
            cur = enc(cur)
            skips.append(cur)
            cur = pool(cur)
        cur = self.bottleneck(cur)
        for up, dec, skip in zip(self.upconvs, self.decs, reversed(skips[:-1] + [skips[-1]])):
            cur = up(cur)
            if self.use_skip:
                # align length if needed (due to odd sizes)
                if skip.shape[-1] != cur.shape[-1]:
                    diff = skip.shape[-1] - cur.shape[-1]
                    if diff > 0:
                        skip = skip[..., :cur.shape[-1]]
                    else:
                        cur = cur[..., :skip.shape[-1]]
                cur = torch.cat([cur, skip], dim=1)
            cur = dec(cur)
        out = self.out_conv(cur)
        # residual add (predict residual)
        # ensure same shape as input
        if out.shape != x.shape:
            # center crop/pad out
            min_len = min(out.shape[-1], x.shape[-1])
            out = out[..., :min_len]
        out = x + out  # residual connection: denoised = input + predicted_residual
        return out
