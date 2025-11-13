#!/usr/bin/env python3
"""
eval_and_reports.py
Run denoiser on a test folder of paired wavs and produce CSV + plots.

Usage:
  python eval_and_reports.py --checkpoint checkpoints/best.pt --data_root /path/to/dataset_root --out_dir reports --device cuda
"""
import argparse, os, csv
from glob import glob
import numpy as np
import torch
import torchaudio
from pinn_audio.model import UNet1D
from pinn_audio.data import load_audio
from pinn_audio.physics import physics_residual, compute_dimensionless, compute_dt_from_tprime
import matplotlib.pyplot as plt
from tqdm import tqdm

# reuse SI-SDR and SNR functions (same as earlier)
def si_sdr(est, ref, eps=1e-8):
    ref = ref - ref.mean(); est = est - est.mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return 10*np.log10((np.sum(target**2)+eps)/(np.sum(noise**2)+eps))

def snr(est, ref, eps=1e-8):
    sig = np.sum(ref**2); noise = np.sum((ref - est)**2)
    return 10*np.log10((sig+eps)/(noise+eps))

def make_windows(x, win_len, hop_len):
    T = x.shape[-1]
    starts = list(range(0, max(1, T - win_len + 1), hop_len))
    if len(starts) == 0: starts=[0]
    arr=[]
    for s in starts:
        w = np.zeros(win_len,dtype=np.float32)
        avail=min(win_len, T-s); w[:avail]=x[s:s+avail]; arr.append(w)
    return np.stack(arr)[:,None,:], starts

def overlap_add(windows, hop_len):
    N,C,L = windows.shape
    total = (N-1)*hop_len + L
    out=np.zeros(total,dtype=np.float32); wt=np.zeros(total,dtype=np.float32); win=np.hanning(L).astype(np.float32)
    for i in range(N):
        s=i*hop_len; out[s:s+L]+=windows[i,0]*win; wt[s:s+L]+=win
    nz = wt>1e-8; out[nz] /= wt[nz]
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="reports")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--win_sec", type=float, default=1.0)
    p.add_argument("--hop_sec", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args=p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    model = UNet1D()
    ck = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ck.get("model_state", ck))
    model.to(device).eval()

    test_dir = os.path.join(args.data_root, "test")
    pairs = []
    for p in glob(os.path.join(test_dir, "*_noisy.wav")):
        base = os.path.basename(p).replace("_noisy.wav","")
        noisy = p
        clean = os.path.join(test_dir, base + "_clean.wav")
        if os.path.exists(clean):
            pairs.append((base, noisy, clean))
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "results.csv")
    rows = [["utt", "mse_noisy", "mse_denoised", "si_sdr_noisy", "si_sdr_denoised", "snr_noisy", "snr_denoised", "phys_res_mean"]]
    phys_means = []

    win_len = int(args.win_sec*args.sr); hop_len=int(args.hop_sec*args.sr)
    for uid, noisy_p, clean_p in tqdm(pairs):
        wn, _ = load_audio(noisy_p, args.sr); wc, _ = load_audio(clean_p, args.sr)
        wn = wn.squeeze(0).numpy().astype(np.float32)
        wc = wc.squeeze(0).numpy().astype(np.float32)
        wins, _ = make_windows(wn, win_len, hop_len)
        den_wins=[]
        with torch.no_grad():
            for i in range(0, wins.shape[0], 8):
                xb = torch.from_numpy(wins[i:i+8]).to(device)
                out = model(xb).detach().cpu().numpy()
                den_wins.append(out)
        den_wins = np.concatenate(den_wins, axis=0)
        recon = overlap_add(den_wins, hop_len)[:len(wn)]

        mse_noisy = float(np.mean((wn - wc)**2))
        mse_den = float(np.mean((recon - wc)**2))
        si_no = float(si_sdr(wn, wc))
        si_de = float(si_sdr(recon, wc))
        sn_no = float(snr(wn, wc)); sn_de = float(snr(recon, wc))

        # physics residual mean (compute on dimensionless p_hat_prime if possible)
        try:
            # compute p_hat_prime similar to training: p_hat / P0
            from pinn_audio.data import PairedWavWindowDataset
            # quick P0: rms of clean window across whole file
            P0 = np.sqrt(np.maximum(1e-8, np.mean(wc**2)))
            p_hat_prime = recon / (P0 + 1e-8)
            # create tprime
            T = len(recon)
            tprime = torch.linspace(0.0, 1.0, steps=T).unsqueeze(0).unsqueeze(0)  # [1,1,T]
            import torch as _torch
            r = physics_residual(_torch.from_numpy(p_hat_prime).unsqueeze(0).unsqueeze(0).float(), tprime, 0.2, (2*3.14159*300*args.win_sec)**2)
            phys_mean = float(r.abs().mean().item())
        except Exception as e:
            phys_mean = float('nan')

        phys_means.append(phys_mean)
        rows.append([uid, mse_noisy, mse_den, si_no, si_de, sn_no, sn_de, phys_mean])

        # save example overlays for first few
        if len(rows) <= 6:
            import matplotlib.pyplot as plt
            t = np.arange(len(wn))/args.sr
            plt.figure(figsize=(10,3)); plt.plot(t, wn, label='noisy', alpha=0.6); plt.plot(t, recon, label='den', alpha=0.8); plt.plot(t, wc, label='clean', alpha=0.8); plt.legend(); plt.title(uid); plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, f"{uid}_overlay.png")); plt.close()

    # write csv
    with open(csv_path, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerows(rows)
    print("Wrote", csv_path)

    # stats
    phys_means = np.array([x for x in phys_means if not np.isnan(x)])
    plt.figure(figsize=(6,4)); plt.hist(phys_means, bins=50); plt.title("Physics residual mean histogram"); plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "phys_res_hist.png"))

    # scatter: si_sdr improvement
    data = np.array(rows[1:])
    si_no = data[:,3].astype(float); si_de = data[:,4].astype(float)
    plt.figure(figsize=(6,6)); plt.scatter(si_no, si_de, alpha=0.5); plt.plot([-20,40], [-20,40], 'r--'); plt.xlabel("SI-SDR noisy"); plt.ylabel("SI-SDR denoised"); plt.savefig(os.path.join(args.out_dir, "si_scatter.png"))

    print("Reports saved to", args.out_dir)

if __name__ == "__main__":
    main()
