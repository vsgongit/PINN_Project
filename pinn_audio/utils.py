"""
pinn_audio.utils
Utility functions: seeding, save/load checkpoint, audio IO, metrics, plotting helpers
"""

import os
import random
import torch
import numpy as np
import torchaudio
import csv
from typing import Dict, Tuple, List

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic if desired
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device):
    return torch.load(path, map_location=device)


def write_metrics_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_wav(path: str, waveform: torch.Tensor, sr: int = 16000):
    """
    waveform: [1,T] or [T], float32 in [-1,1]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform.cpu(), sr)
