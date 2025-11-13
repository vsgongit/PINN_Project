"""
Training loop for PINN audio denoiser (robust, safe).
This file replaces the broken version and includes robust handling for tprime shapes.
"""
import os
import math
from tqdm import tqdm
import time
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from pinn_audio.data import PairedWavWindowDataset
from pinn_audio.model import UNet1D
from pinn_audio.physics import compute_dimensionless, physics_residual, physics_loss_from_residual
from pinn_audio.losses import mse_loss, si_sdr, stft_mag_loss
from pinn_audio.utils import set_seed, save_checkpoint, load_checkpoint

def make_dataloaders(data_root, csv_map, sr, win_sec, hop_sec, batch_size, num_workers):
    train_ds = PairedWavWindowDataset(root=data_root, csv_map=csv_map, subset="train", sr=sr, win_sec=win_sec, hop_sec=hop_sec)
    val_ds = PairedWavWindowDataset(root=data_root, csv_map=csv_map, subset="val", sr=sr, win_sec=win_sec, hop_sec=hop_sec)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def validate_and_checkpoint(model, val_loader, device, writer, epoch, checkpoint_dir):
    model.eval()
    total_si = 0.0
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for noisy, clean, p0_batch, tprime, dt in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            batch_si = si_sdr(pred.cpu(), clean.cpu()).item()
            total_si += batch_si
            total_mse += F.mse_loss(pred, clean).item()
            count += 1
    avg_si = total_si / max(1, count)
    avg_mse = total_mse / max(1, count)
    if writer:
        writer.add_scalar("val/si_sdr", avg_si, epoch)
        writer.add_scalar("val/mse", avg_mse, epoch)
    print(f"Validation epoch {epoch}: SI-SDR={avg_si:.3f} dB, MSE={avg_mse:.6f}")
    return avg_si

def train(
    data_root: Optional[str],
    csv_map: Optional[str],
    sr: int = 16000,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 2e-4,
    lambda_phys: float = 0.2,
    lambda_stft: float = 0.0,
    alpha_prime_val: float = 0.2,
    beta_prime_val: float = None,
    learn_alpha: bool = False,
    learn_beta: bool = False,
    checkpoint_dir: str = "checkpoints",
    seed: int = 1234,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 0,
    log_dir: str = "runs",
    save_every: int = 1,
    warmup_epochs: int = 2,
):
    set_seed(seed)
    if beta_prime_val is None:
        beta_prime_val = (2.0 * math.pi * 300.0 * win_sec) ** 2

    train_loader, val_loader = make_dataloaders(data_root, csv_map, sr, win_sec, hop_sec, batch_size, num_workers)

    model = UNet1D().to(device)

    # physics params (fixed or learnable via softplus)
    if learn_alpha:
        alpha_param = nn.Parameter(torch.tensor(alpha_prime_val, device=device))
    else:
        alpha_param = torch.tensor(alpha_prime_val, device=device)

    if learn_beta:
        beta_param = nn.Parameter(torch.tensor(beta_prime_val, device=device))
    else:
        beta_param = torch.tensor(beta_prime_val, device=device)

    params = list(model.parameters())
    if learn_alpha:
        params.append(alpha_param)
    if learn_beta:
        params.append(beta_param)

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs

    def lr_lambda(step):
        if step < warmup_epochs * len(train_loader):
            return float(step) / float(max(1, warmup_epochs * len(train_loader)))
        progress = float(step - warmup_epochs * len(train_loader)) / float(max(1, total_steps - warmup_epochs * len(train_loader)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.amp.GradScaler(enabled=(device.startswith("cuda")))
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    best_val_si = -1e9
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running = {"data": 0.0, "phys": 0.0, "stft": 0.0, "total": 0.0}
        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}/{epochs}")
        for noisy, clean, p0, tprime, dt in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            p0 = p0.to(device)

            # Robust handling of tprime shape variations:
            # possible shapes: [1,1,T], [B,1,1,T], [B,1,T]
            tprime = tprime.to(device)
            if tprime.dim() == 4 and tprime.shape[2] == 1:
                tprime = tprime.squeeze(2).clone().requires_grad_(True)  # [B,1,T]
            elif tprime.dim() == 3 and tprime.shape[0] == 1:
                tprime = tprime.expand(noisy.shape[0], -1, -1).clone().requires_grad_(True)
            else:
                tprime = tprime.clone().requires_grad_(True)

            with torch.autocast(device_type='cuda' if device.startswith("cuda") else 'cpu', enabled=(device.startswith("cuda"))):
                pred = model(noisy)
                loss_data = mse_loss(pred, clean)
                # physics loss: use softplus if learnable, else fixed
                if learn_alpha:
                    alpha_prime = F.softplus(alpha_param)
                else:
                    alpha_prime = alpha_param
                if learn_beta:
                    beta_prime = F.softplus(beta_param)
                else:
                    beta_prime = beta_param

                p_hat_prime = pred / (p0.unsqueeze(-1) + 1e-8)
                residual = physics_residual(p_hat_prime, tprime, alpha_prime, beta_prime)
                loss_phys = physics_loss_from_residual(residual)
                loss_stft = torch.tensor(0.0, device=device)
                if lambda_stft > 0.0:
                    loss_stft = stft_mag_loss(pred, clean)
                loss = loss_data + lambda_phys * loss_phys + lambda_stft * loss_stft

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            running["data"] += loss_data.item()
            running["phys"] += loss_phys.item()
            running["stft"] += (loss_stft.item() if torch.is_tensor(loss_stft) else 0.0)
            running["total"] += loss.item()
            pbar.set_postfix({
                "ld": running["data"]/global_step if global_step>0 else 0.0,
                "lp": running["phys"]/global_step if global_step>0 else 0.0,
                "lt": running["total"]/global_step if global_step>0 else 0.0,
                "lr": scheduler.get_last_lr()[0]
            })

        if writer:
            writer.add_scalar("train/loss_data", running["data"]/len(train_loader), epoch)
            writer.add_scalar("train/loss_phys", running["phys"]/len(train_loader), epoch)
            writer.add_scalar("train/loss_total", running["total"]/len(train_loader), epoch)

        val_si = validate_and_checkpoint(model, val_loader, device, writer, epoch, checkpoint_dir)

        if val_si > best_val_si:
            best_val_si = val_si
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_si": val_si
            }, os.path.join(checkpoint_dir, "best.pt"))

        if (epoch + 1) % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))

    if writer:
        writer.close()
