# PINN_Project
Audio denoiser using PINN

# PINN Audio Denoiser (Temporal PINN, PyTorch)

Physics-informed waveform denoiser using temporal ODE residuals as a physics loss.

## Requirements

- Python 3.10+
- PyTorch 2.x (+ CUDA optionally)
- torchaudio
- numpy
- scipy
- matplotlib
- tensorboard
- librosa (for evaluation plots)
- soundfile (optional)

Install:
```bash
pip install torch torchvision torchaudio   # follow official instructions for CUDA
pip install numpy scipy matplotlib tensorboard librosa soundfile