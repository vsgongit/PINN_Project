"""
pinn_audio.eval
Run evaluation on test split, compute MSE, MAE, SI-SDR, ΔSNR, save denoised WAVs, and plots.
"""

import os
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from pinn_audio.data import PairedWavWindowDataset
from pinn_audio.model import UNet1D
from pinn_audio.losses import si_sdr
from pinn_audio.utils import save_wav, load_checkpoint, write_metrics_csv


def eval_and_export(checkpoint_path: str, data_root: str, csv_map: str = None, out_dir: str = "outputs", sr: int = 16000, win_sec: float = 1.0, hop_sec: float = 0.5, batch_size: int = 8, device="cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device)
    # load model
    ckpt = load_checkpoint(checkpoint_path, device)
    model = UNet1D()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    test_ds = PairedWavWindowDataset(root=data_root, csv_map=csv_map, subset="test", sr=sr, win_sec=win_sec, hop_sec=hop_sec)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    metrics_rows = []
    example_plots = []
    total_si_pred = 0.0
    total_si_noisy = 0.0
    count = 0
    for noisy, clean, p0, tprime, dt in tqdm(loader, desc="Eval"):
        noisy = noisy.to(device)
        clean = clean.to(device)
        with torch.no_grad():
            pred = model(noisy)
        # per-example metrics
        for i in range(pred.shape[0]):
            pred_i = pred[i].detach().cpu().squeeze(0)
            noisy_i = noisy[i].detach().cpu().squeeze(0)
            clean_i = clean[i].detach().cpu().squeeze(0)
            # compute SI-SDR
            si_pred = si_sdr(pred[i:i+1].cpu(), clean[i:i+1].cpu()).item()
            si_noisy = si_sdr(noisy[i:i+1].cpu(), clean[i:i+1].cpu()).item()
            total_si_pred += si_pred
            total_si_noisy += si_noisy
            count += 1
            # save wav
            out_subdir = os.path.join(out_dir, "wavs")
            os.makedirs(out_subdir, exist_ok=True)
            idx_name = f"example_{count:05d}"
            save_wav(os.path.join(out_subdir, f"{idx_name}_noisy.wav"), noisy_i, sr)
            save_wav(os.path.join(out_subdir, f"{idx_name}_pred.wav"), pred_i, sr)
            save_wav(os.path.join(out_subdir, f"{idx_name}_clean.wav"), clean_i, sr)
            metrics_rows.append({
                "id": idx_name,
                "si_pred_db": si_pred,
                "si_noisy_db": si_noisy,
                "si_improve_db": si_pred - si_noisy,
                "mse": float(((pred_i - clean_i) ** 2).mean().item())
            })
            # store a few examples for plotting
            if len(example_plots) < 3:
                example_plots.append((noisy_i.numpy(), pred_i.numpy(), clean_i.numpy()))
    # write csv
    write_metrics_csv(os.path.join(out_dir, "metrics.csv"), metrics_rows)
    print(f"Avg SI-SDR (pred): {total_si_pred / count:.3f} dB; Avg SI-SDR (noisy): {total_si_noisy / count:.3f} dB; ΔSI = {(total_si_pred - total_si_noisy)/count:.3f} dB")
    # plots
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for i, (noisy_v, pred_v, clean_v) in enumerate(example_plots):
        T = noisy_v.shape[-1]
        t = np.linspace(0, win_sec, T)
        plt.figure(figsize=(10, 6))
        plt.plot(t, noisy_v, label="noisy", alpha=0.6)
        plt.plot(t, pred_v, label="denoised", alpha=0.8)
        plt.plot(t, clean_v, label="clean", alpha=0.8)
        plt.legend()
        plt.title(f"Example {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"time_example_{i}.png"))
        plt.close()
        # spectrograms
        import librosa
        import librosa.display
        S_noisy = librosa.stft(noisy_v, n_fft=512, hop_length=128)
        S_pred = librosa.stft(pred_v, n_fft=512, hop_length=128)
        S_clean = librosa.stft(clean_v, n_fft=512, hop_length=128)
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_noisy), ref=np.max), sr=sr, hop_length=128, x_axis='time', y_axis='hz')
        plt.title('Noisy (log-mag)')
        plt.colorbar(format="%+2.0f dB")
        plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_pred), ref=np.max), sr=sr, hop_length=128, x_axis='time', y_axis='hz')
        plt.title('Predicted (log-mag)')
        plt.colorbar(format="%+2.0f dB")
        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_clean), ref=np.max), sr=sr, hop_length=128, x_axis='time', y_axis='hz')
        plt.title('Clean (log-mag)')
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"spec_example_{i}.png"))
        plt.close()
