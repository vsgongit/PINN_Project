import argparse, os, shutil, random, csv
from glob import glob
from pathlib import Path

def list_wavs(folder):
    files = []
    for e in ("*.wav", "*.WAV"):
        files.extend(glob(os.path.join(folder, e)))
    return sorted(files)

def basename_no_ext(p): return Path(p).stem

def main(args):
    noisy_files = list_wavs(args.noisy_dir)
    clean_files = list_wavs(args.clean_dir)
    print(f"Found {len(noisy_files)} noisy and {len(clean_files)} clean files")

    noisy_map = {basename_no_ext(p): p for p in noisy_files}
    clean_map = {basename_no_ext(p): p for p in clean_files}
    common = sorted(set(noisy_map) & set(clean_map))
    if not common:
        raise RuntimeError("No matching basenames found.")

    pairs = [(noisy_map[b], clean_map[b], b) for b in common]
    random.seed(args.seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train, n_val = int(n*args.split[0]), int(n*args.split[1])
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train+n_val],
        "test": pairs[n_train+n_val:],
    }

    out_root = Path(args.out_root)
    for subset, plist in splits.items():
        out_dir = out_root/subset
        out_dir.mkdir(parents=True, exist_ok=True)
        for noisy_path, clean_path, uid in plist:
            n_dst = out_dir/f"{uid}_noisy.wav"
            c_dst = out_dir/f"{uid}_clean.wav"
            if args.mode=="copy":
                shutil.copy2(noisy_path, n_dst)
                shutil.copy2(clean_path, c_dst)
            else:
                if n_dst.exists(): n_dst.unlink()
                if c_dst.exists(): c_dst.unlink()
                try:
                    os.symlink(os.path.abspath(noisy_path), n_dst)
                    os.symlink(os.path.abspath(clean_path), c_dst)
                except OSError:
                    shutil.copy2(noisy_path, n_dst)
                    shutil.copy2(clean_path, c_dst)

    # write mapping CSV
    with open(out_root/"mapping.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["utt_id","noisy_path","clean_path"])
        for n,c,u in pairs: w.writerow([u,os.path.abspath(n),os.path.abspath(c)])
    print("Dataset prepared at", out_root)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--noisy_dir",required=True)
    p.add_argument("--clean_dir",required=True)
    p.add_argument("--out_root",required=True)
    p.add_argument("--split",nargs=3,type=float,default=[0.15,0.15,0.7])
    p.add_argument("--mode",choices=["copy","symlink"],default="copy")
    p.add_argument("--seed",type=int,default=1234)
    main(p.parse_args())