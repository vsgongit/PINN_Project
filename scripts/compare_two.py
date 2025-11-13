#!/usr/bin/env python3
"""
compare_two.py
Run denoiser on two input files (can be two noisy files or noisy+clean pairs),
compute metrics (MSE, SI-SDR, SNR, ΔSNR), and plot overlays & spectrograms.

Usage:
  python compare_two.py --checkpoint ckpt.pt --a a_noisy.wav --b b_noisy.wav --clean_a a_clean.wav --clean_b b_clean.wav
"""
import argparse
import numpy as np
import torchaudio
import torch
from pinn_audio.model import UNet1D
from pinn_audio.data import load_audio
import matplotlib.pyplot as plt
import math

# SI-SDR implementation (simple)
def si_sdr(est, ref, eps=1e-8):
    # est, ref: numpy 1D arrays
    ref = ref.astype(np.float32)
    est = est.astype(np.float32)
    def zero_mean(x): return x - np.mean(x)
    r = zero_mean(ref)
    e = zero_mean(est)
    alpha = np.dot(r, e) / (np.dot(r, r) + eps)
    target = alpha * r
    noise = e - target
    return 10 * np.log10((np.sum(target**2) + eps) / (np.sum(noise**2) + eps))

def snr(est, ref, eps=1e-8):
    signal = np.sum(ref**2)
    noise = np.sum((ref - est)**2)
    return 10 * np.log10((signal + eps) / (noise + eps))

def run_one(model, device, inp, sr, win_sec=1.0, hop_sec=0.5):
    from infer_clip import make_windows, overlap_add
    win_len = int(win_sec * sr); hop_len = int(hop_sec * sr)
    windows, _ = make_windows(inp, win_len, hop_len)
    denoised_windows = []
    with torch.no_grad():
        for i in range(0, windows.shape[0], 8):
            xb = torch.from_numpy(windows[i:i+8]).to(device)
            out = model(xb).detach().cpu().numpy()
            denoised_windows.append(out)
    den = np.concatenate(denoised_windows, axis=0)
    recon = overlap_add(den, hop_len)[:len(inp)]
    return recon

def plot_pair(a_noisy, a_denoised, a_clean, b_noisy, b_denoised, b_clean, sr, out_prefix):
    t = np.arange(len(a_noisy))/sr
    plt.figure(figsize=(12,3)); plt.plot(t,a_noisy,label='noisy'); plt.plot(t,a_denoised,label='den'); plt.plot(t,a_clean,label='clean'); plt.legend(); plt.title("A (time)"); plt.savefig(out_prefix + "_A_time.png"); plt.close()
    t2 = np.arange(len(b_noisy))/sr
    plt.figure(figsize=(12,3)); plt.plot(t2,b_noisy,label='noisy'); plt.plot(t2,b_denoised,label='den'); plt.plot(t2,b_clean,label='clean'); plt.legend(); plt.title("B (time)"); plt.savefig(out_prefix + "_B_time.png"); plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--clean_a", default=None)
    p.add_argument("--clean_b", default=None)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    model = UNet1D()
    ck = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ck.get("model_state", ck))
    model.to(device).eval()

    wa, _ = load_audio(args.a, args.sr); wb, _ = load_audio(args.b, args.sr)
    wa = wa.squeeze(0).numpy().astype(np.float32); wb = wb.squeeze(0).numpy().astype(np.float32)
    da = run_one(model, device, wa, args.sr); db = run_one(model, device, wb, args.sr)

    ca = None; cb = None
    if args.clean_a: ca = load_audio(args.clean_a, args.sr)[0].squeeze(0).numpy()
    if args.clean_b: cb = load_audio(args.clean_b, args.sr)[0].squeeze(0).numpy()

    # metrics
    print("A metrics:")
    if ca is not None:
        print("  SI-SDR noisy:", si_sdr(wa, ca), "denoised:", si_sdr(da, ca))
        print("  SNR noisy:", snr(wa, ca), "denoised:", snr(da, ca))
    print("  MSE noisy->clean:", np.mean((wa - (ca if ca is not None else da))**2))
    print("B metrics:")
    if cb is not None:
        print("  SI-SDR noisy:", si_sdr(wb, cb), "denoised:", si_sdr(db, cb))
        print("  SNR noisy:", snr(wb, cb), "denoised:", snr(db, cb))

    out_pref = "compare_out"
    plot_pair(wa, da, ca if ca is not None else np.zeros_like(da), wb, db, cb if cb is not None else np.zeros_like(db), args.sr, out_pref)
    print("Plots saved with prefix:", out_pref)

if __name__ == "__main__":
    main()
