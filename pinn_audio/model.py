"""
Simple length-preserving UNet1D for audio denoising.
- Input: [B,1,T]
- Output: [B,1,T] (residual predicted, added to input)
This implementation uses convs with padding=(k-1)//2 and ConvTranspose1d
with output_padding when necessary to preserve lengths.
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch, kernel=15, stride=1, padding=None):
    if padding is None:
        padding = (kernel - 1) // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm1d(out_ch),
        nn.SiLU()
    )

def convt_block(in_ch, out_ch, kernel=15, stride=2, padding=None, output_padding=0):
    if padding is None:
        padding = (kernel - 1) // 2
    return nn.Sequential(
        nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding),
        nn.BatchNorm1d(out_ch),
        nn.SiLU()
    )

class UNet1D(nn.Module):
    def __init__(self, widths: List[int] = None, kernel: int = 15):
        super().__init__()
        if widths is None:
            widths = [32, 64, 128, 256]
        self.kernel = kernel

        # encoder
        self.enc_convs = nn.ModuleList()
        prev = 1
        for w in widths:
            # downsample by stride=2 conv
            self.enc_convs.append(conv_block(prev, w, kernel=kernel, stride=2))
            prev = w

        # bottleneck
        self.bottleneck = nn.Sequential(
            conv_block(prev, prev, kernel=kernel, stride=1),
            conv_block(prev, prev, kernel=kernel, stride=1),
        )

        # decoder (mirror)
        self.dec_convs = nn.ModuleList()
        rev = list(reversed(widths))
        for i, w in enumerate(rev[:-1]):
            # use ConvTranspose1d with stride=2 to upsample
            # output_padding chosen as 1 when needed to match odd/even lengths
            self.dec_convs.append(convt_block(rev[i], rev[i+1], kernel=kernel, stride=2, output_padding=0))
        # last decoder to go back to smallest width
        self.dec_convs.append(convt_block(rev[-1], widths[0], kernel=kernel, stride=2, output_padding=0))

        # final conv to single channel residual
        self.out_conv = nn.Conv1d(widths[0], 1, kernel_size=1)

        # small projection if channels need match
        self.out_proj = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x):
        # x: [B,1,T]
        enc_feats = []
        cur = x
        # Encoder
        for conv in self.enc_convs:
            cur = conv(cur)   # downsampled by 2 each time
            enc_feats.append(cur)

        # Bottleneck
        cur = self.bottleneck(cur)

        # Decoder (mirror) — pop encoder features for skip connections
        for i, deconv in enumerate(self.dec_convs):
            # corresponding skip (from reversed order)
            skip_idx = len(enc_feats) - 1 - i
            cur = deconv(cur)
            if skip_idx >= 0:
                skip = enc_feats[skip_idx]
                # if lengths mismatch by 1 due to odd/even, trim or pad
                t_cur = cur.shape[-1]
                t_skip = skip.shape[-1]
                if t_cur > t_skip:
                    cur = cur[:, :, :t_skip]
                elif t_cur < t_skip:
                    # pad cur to match skip
                    pad_amt = t_skip - t_cur
                    cur = F.pad(cur, (0, pad_amt))
                # channel-wise concat
                # if channels differ, project to same channels by 1x1 conv (simple approach)
                if cur.shape[1] != skip.shape[1]:
                    # reduce both to min channels by 1x1 conv if needed
                    min_ch = min(cur.shape[1], skip.shape[1])
                    cur = cur[:, :min_ch, :]
                    skip = skip[:, :min_ch, :]
                cur = cur + skip  # simple residual-style fusion

        # project back to single channel residual
        out = self.out_conv(cur)
        # ensure same time length as input
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode="linear", align_corners=False)
        # final residual add
        res = out
        # if channels mismatch, make both single-channel
        if x.shape[1] != res.shape[1]:
            if res.shape[1] != 1:
                res = res.mean(dim=1, keepdim=True)
            if x.shape[1] != 1:
                # make input single channel by mean
                x_in = x.mean(dim=1, keepdim=True)
            else:
                x_in = x
            return x_in + res
        return x + res
