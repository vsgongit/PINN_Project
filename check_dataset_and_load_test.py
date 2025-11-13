# check_dataset_and_load_test.py
import os, sys
from glob import glob

ROOT = r".\dataset_root"   # run from D:\git_repos\PINN_Project
train_dir = os.path.join(ROOT, "train")

noisy = sorted(glob(os.path.join(train_dir, "*_noisy.wav")))
clean = sorted(glob(os.path.join(train_dir, "*_clean.wav")))

print("train dir:", train_dir)
print("count noisy:", len(noisy))
print("count clean:", len(clean))
print("first noisy sample:", noisy[0] if noisy else None)
print("first clean sample:", clean[0] if clean else None)

# Try loading the first noisy file with torchaudio and soundfile
if noisy:
    p = noisy[0]
    print("\n=== try torchaudio.load ===")
    try:
        import torchaudio
        w, sr = torchaudio.load(p)
        print("torchaudio loaded:", w.shape, "sr=", sr)
    except Exception as e:
        print("torchaudio failed:", repr(e))

    print("\n=== try soundfile (pysoundfile) ===")
    try:
        import soundfile as sf
        data, sr2 = sf.read(p)
        import numpy as np
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        else:
            data = data.T
        print("soundfile loaded:", data.shape, "sr=", sr2)
    except Exception as e:
        print("soundfile failed:", repr(e))

    print("\n=== try librosa.load ===")
    try:
        import librosa
        y, sr3 = librosa.load(p, sr=None, mono=True)
        import numpy as np
        print("librosa loaded:", y.shape, "sr=", sr3)
    except Exception as e:
        print("librosa failed:", repr(e))
else:
    print("No noisy files to test.")
