## Open PowerShell in VS Code’s Terminal
1. Setup the Project Environment

cd D:\git_repos\PINN_Project

2. Activate Virtual Environment

.\.venv\Scripts\Activate.ps1

3. Install Dependencies (First Time Only)

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy<2" soundfile librosa soxr tqdm scipy matplotlib tensorboard

4. Prepare Dataset (Only Needed When Creating dataset_root)

if paths are -> "D:\ml_project\dataset1\noisy_testset_wav"
                "D:\ml_project\dataset1\clean_testset_wav"

python prepare_dataset.py --noisy_dir "D:\ml_project\dataset1\noisy_testset_wav" --clean_dir "D:\ml_project\dataset1\clean_testset_wav" --out_root ".\dataset_root" --split 0.15 0.15 0.7 --mode copy

5. Optional: Verify Dataset is Working

python -c "from pinn_audio.data import PairedWavWindowDataset; ds=PairedWavWindowDataset(root='./dataset_root', subset='train', sr=16000, win_sec=1.0, hop_sec=0.5); print('Registry length:', len(ds))"

6. Train the Model

python -m pinn_audio.cli train --data_root .\dataset_root --batch_size 8 --epochs 10 --num_workers 4 --lambda_phys 0.2
