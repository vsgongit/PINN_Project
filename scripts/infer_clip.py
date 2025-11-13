#!/usr/bin/env python3
"""
infer_clip.py
Load checkpoint, denoise single WAV (sliding-window overlap-add), save output,
and show quick plots (time waveform + log-spectrogram).

Usage:
  python infer_clip.py \
    --checkpoint checkpoints/best.pt \
    --input noisy.wav \
    --output denoised.wav \
    --sr 16000 --win_sec 1.0 --hop_sec 0.5 --device cuda
"""
import argparse
from pathlib import Path
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pinn_audio.model import UNet1D
from pinn_audio.data import load_audio

def make_windows(x, win_len, hop_len):
    T = x.shape[-1]
    starts = list(range(0, max(1, T - win_len + 1), hop_len))
    if len(starts) == 0:
        starts = [0]
    wlist = []
    for s in starts:
        w = np.zeros(win_len, dtype=np.float32)
        avail = min(win_len, T - s)
        w[:avail] = x[s:s+avail]
        wlist.append(w)
    arr = np.stack(wlist, axis=0)[:, np.newaxis, :]
    return arr, starts

def overlap_add(windows, hop_len):
    N, C, L = windows.shape
    total_len = (N - 1) * hop_len + L
    out = np.zeros(total_len, dtype=np.float32)
    weight = np.zeros(total_len, dtype=np.float32)
    win = np.hanning(L).astype(np.float32)
    for i in range(N):
        s = i * hop_len
        out[s:s+L] += windows[i,0] * win
        weight[s:s+L] += win
    nz = weight > 1e-8
    out[nz] /= weight[nz]
    return out

def plot_waveforms(t, noisy, denoised, clean=None, title=None, out_png=None):
    plt.figure(figsize=(12,4))
    plt.plot(t, noisy, label="noisy", alpha=0.6)
    plt.plot(t, denoised, label="denoised", alpha=0.8)
    if clean is not None:
        plt.plot(t, clean, label="clean", alpha=0.8)
    plt.legend()
    plt.title(title or "Waveforms")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
    else:
        plt.show()
    plt.close()

def plot_spectrogram(x, sr, title=None, out_png=None):
    S = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0/sr)
    plt.figure(figsize=(10,4))
    plt.semilogy(freqs, S + 1e-8)
    plt.title(title or "Log-spectrum")
    plt.xlabel("Hz")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
    else:
        plt.show()
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--clean", default=None, help="optional clean reference for plots/metrics")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--win_sec", type=float, default=1.0)
    p.add_argument("--hop_sec", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # load model + weights
    model = UNet1D()
    ck = torch.load(args.checkpoint, map_location="cpu")
    state = ck.get("model_state", ck)
    model.load_state_dict(state)
    model.to(device).eval()

    # load wav
    wav, _sr = load_audio(args.input, args.sr)
    wav = wav.squeeze(0).numpy().astype(np.float32)

    win_len = int(args.win_sec * args.sr)
    hop_len = int(args.hop_sec * args.sr)
    windows, starts = make_windows(wav, win_len, hop_len)

    denoised_windows = []
    batch = 8
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch):
            xb = torch.from_numpy(windows[i:i+batch]).to(device)
            out = model(xb)  # model should return denoised waveform (input + residual) if designed so
            out = out.detach().cpu().numpy()
            denoised_windows.append(out)
    denoised = np.concatenate(denoised_windows, axis=0)
    recon = overlap_add(denoised, hop_len)[:len(wav)]

    # save
    torchaudio.save(args.output, torch.from_numpy(recon).unsqueeze(0), args.sr)
    print("Saved denoised audio ->", args.output)

    # optional plotting
    t = np.arange(len(wav))/args.sr
    clean = None
    if args.clean:
        cw, _ = load_audio(args.clean, args.sr); clean = cw.squeeze(0).numpy().astype(np.float32)
    plot_waveforms(t, wav, recon, clean=clean, title="Noisy vs Denoised vs Clean")
    plot_spectrogram(wav, args.sr, title="Noisy spectrum")
    plot_spectrogram(recon, args.sr, title="Denoised spectrum")

if __name__ == "__main__":
    main()
