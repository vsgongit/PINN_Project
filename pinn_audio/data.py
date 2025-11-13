"""
Robust dataset for paired noisy/clean wavs with windowing and resampling.
"""
import os
import csv
from glob import glob
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np

def read_csv_map(csv_path: str) -> List[Tuple[str, str, str]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
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
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
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
        min_rms: float = 1e-8,
    ):
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

        self.registry = []
        for utt, noisy_path, clean_path in self.pairs:
            # Try to probe file info (fast) and convert to target-sr sample count
            try:
                info = torchaudio.info(noisy_path)
                num_frames = int(info.num_frames)
                orig_sr = int(info.sample_rate)
                duration_at_target = int(round(num_frames * (self.sr / float(orig_sr))))
            except Exception:
                # fallback: load & resample to know exact length
                try:
                    w, sr0 = load_audio(noisy_path, self.sr)
                    duration_at_target = w.shape[1]
                except Exception as e:
                    print(f"Warning: cannot probe/load {noisy_path}: {e}; skipping")
                    continue

            starts = list(range(0, max(1, duration_at_target - self.win_len + 1), self.hop_len))
            if len(starts) == 0:
                starts = [0]
            for s in starts:
                self.registry.append((noisy_path, clean_path, s, duration_at_target))

        if len(self.registry) == 0:
            raise RuntimeError("No audio windows found; check dataset paths and file naming.")

    def __len__(self):
        return len(self.registry)

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(torch.mean(x ** 2, dim=-1, keepdim=True), min=self.min_rms))

    def __getitem__(self, idx):
        noisy_path, clean_path, start, duration = self.registry[idx]
        noisy, _ = load_audio(noisy_path, self.sr)  # [1, T]
        clean, _ = load_audio(clean_path, self.sr)

        if noisy.ndim == 1:
            noisy = noisy.unsqueeze(0)
        if clean.ndim == 1:
            clean = clean.unsqueeze(0)

        # create zero windows and safely copy available samples
        noisy_win = torch.zeros(1, self.win_len, dtype=noisy.dtype)
        clean_win = torch.zeros(1, self.win_len, dtype=clean.dtype)

        T_noisy = noisy.shape[1] if noisy.numel() > 0 else 0
        T_clean = clean.shape[1] if clean.numel() > 0 else 0

        if start < T_noisy:
            avail = min(self.win_len, T_noisy - start)
            if avail > 0:
                noisy_win[:, :avail] = noisy[:, start:start+avail]
        # else leave zeros

        if start < T_clean:
            avail_c = min(self.win_len, T_clean - start)
            if avail_c > 0:
                clean_win[:, :avail_c] = clean[:, start:start+avail_c]

        # compute P0 (RMS of clean window fallback to noisy)
        p0 = self._rms(clean_win)
        if p0 <= self.min_rms:
            p0 = self._rms(noisy_win)

        # t' grid: [1,1,T]
        T = self.win_len
        tprime = torch.linspace(0.0, 1.0, steps=T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        dt = 1.0 / self.sr
        return noisy_win.clone().float(), clean_win.clone().float(), p0.squeeze(0).unsqueeze(0).float(), tprime, dt
