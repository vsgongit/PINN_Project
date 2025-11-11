"""
pinn_audio.data
Dataset and dataloader utilities.

- Loads WAVs from directory or CSV map
- Resamples to sr (default 16000), converts to mono
- Slices into fixed windows with hop, pads last window
- Returns tensors:
    noisy_win [B,1,T], clean_win [B,1,T], P0 [B,1,1], tprime [B,1,T], dt (float)
"""

import os
import csv
from glob import glob
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np


def read_csv_map(csv_path: str) -> List[Tuple[str, str, str]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Expect either (utt_id, noisy_path, clean_path) or (noisy_path, clean_path)
        for r in reader:
            if len(r) == 3:
                utt, noisy, clean = r
            elif len(r) == 2:
                noisy, clean = r
                utt = os.path.splitext(os.path.basename(noisy))[0]
            else:
                raise ValueError("CSV map rows must have 2 or 3 columns")
            rows.append((utt, noisy, clean))
    return rows


def find_pairs_in_dir(root: str) -> List[Tuple[str, str, str]]:
    """Scans root/train (or val/test) and finds pairs by <utt>_noisy.wav and <utt>_clean.wav"""
    wavs = glob(os.path.join(root, "*.wav"))
    ids = {}
    for w in wavs:
        bn = os.path.basename(w)
        if bn.endswith("_noisy.wav"):
            utt = bn.replace("_noisy.wav", "")
            ids.setdefault(utt, {})["noisy"] = w
        elif bn.endswith("_clean.wav"):
            utt = bn.replace("_clean.wav", "")
            ids.setdefault(utt, {})["clean"] = w
    pairs = []
    for utt, d in ids.items():
        if "noisy" in d and "clean" in d:
            pairs.append((utt, d["noisy"], d["clean"]))
    return pairs


def load_audio(path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)  # [C, T]
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


class PairedWavWindowDataset(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,
        csv_map: Optional[str] = None,
        subset: str = "train",
        sr: int = 16000,
        win_sec: float = 1.0,
        hop_sec: float = 0.5,
        min_rms: float = 1e-6,
    ):
        """
        If csv_map provided, use it. Otherwise expect directory structure root/subset/ with paired files.
        """
        assert (root is not None) or (csv_map is not None)
        self.sr = sr
        self.win_len = int(win_sec * sr)
        self.hop_len = int(hop_sec * sr)
        self.min_rms = min_rms

        if csv_map:
            self.pairs = read_csv_map(csv_map)
        else:
            assert root is not None
            subset_dir = os.path.join(root, subset)
            self.pairs = find_pairs_in_dir(subset_dir)

        # We'll expand to windows lazily: store file-list, then on __getitem__ create windows.
        # To simplify batching, we will precompute window registry: list of (noisy_path, clean_path, start_sample)
        self.registry = []
        for utt, noisy_path, clean_path in self.pairs:
            noisy_wave, sr0 = torchaudio.load(noisy_path)
            if noisy_wave.shape[0] > 1:
                noisy_wave = torch.mean(noisy_wave, dim=0, keepdim=True)
            duration = noisy_wave.shape[1]
            # compute starts
            starts = list(range(0, max(1, duration - self.win_len + 1), self.hop_len))
            if len(starts) == 0:
                starts = [0]
            for s in starts:
                self.registry.append((noisy_path, clean_path, s, duration))
        if len(self.registry) == 0:
            raise RuntimeError("No audio windows found; check dataset paths.")

    def __len__(self):
        return len(self.registry)

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, T]
        return torch.sqrt(torch.clamp(torch.mean(x ** 2, dim=-1, keepdim=True), min=self.min_rms))

    def __getitem__(self, idx):
        noisy_path, clean_path, start, duration = self.registry[idx]
        noisy, _ = load_audio(noisy_path, self.sr)  # [1, T]
        clean, _ = load_audio(clean_path, self.sr)

        # Extract window with padding if necessary
        end = start + self.win_len
        if end <= noisy.shape[1]:
            noisy_win = noisy[:, start:end]
            clean_win = clean[:, start:end]
        else:
            # pad
            noisy_win = torch.zeros(1, self.win_len)
            clean_win = torch.zeros(1, self.win_len)
            available = noisy.shape[1] - start
            noisy_win[:, :available] = noisy[:, start:]
            clean_win[:, :available] = clean[:, start:]

        # Normalize to [-1,1] already torchaudio loads as float32 typically -1..1 but we ensure scaling if needed
        # compute P0 as RMS of clean (fallback to noisy)
        p0 = self._rms(clean_win)  # [1,1]
        if p0 <= self.min_rms:
            p0 = self._rms(noisy_win)
        # t' grid shape [1, T] values in [0,1]
        T = self.win_len
        tprime = torch.linspace(0.0, 1.0, steps=T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,T]
        dt = 1.0 / self.sr
        return noisy_win.clone().float(), clean_win.clone().float(), p0.squeeze(0).unsqueeze(0).float(), tprime, dt
