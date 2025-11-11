"""
CLI entrypoint for training and evaluation.
"""

import argparse
import os
from pinn_audio.train import train
from pinn_audio.eval import eval_and_export

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    # train
    t = sub.add_parser("train")
    t.add_argument("--data_root", type=str, required=True)
    t.add_argument("--csv_map", type=str, default=None)
    t.add_argument("--sr", type=int, default=16000)
    t.add_argument("--win_sec", type=float, default=1.0)
    t.add_argument("--hop_sec", type=float, default=0.5)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--lambda_phys", type=float, default=0.2)
    t.add_argument("--lambda_stft", type=float, default=0.0)
    t.add_argument("--alpha_prime", type=float, default=0.2)
    t.add_argument("--beta_prime", type=float, default=None)
    t.add_argument("--learn_alpha", action="store_true")
    t.add_argument("--learn_beta", action="store_true")
    t.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--num_workers", type=int, default=4)
    t.add_argument("--log_dir", type=str, default="runs")
    t.add_argument("--save_every", type=int, default=1)
    t.add_argument("--warmup_epochs", type=int, default=2)
    # eval
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", type=str, required=True)
    e.add_argument("--data_root", type=str, required=True)
    e.add_argument("--csv_map", type=str, default=None)
    e.add_argument("--out_dir", type=str, default="outputs")
    e.add_argument("--sr", type=int, default=16000)
    e.add_argument("--win_sec", type=float, default=1.0)
    e.add_argument("--hop_sec", type=float, default=0.5)
    e.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()

def main():
    args = parse_args()
    if args.cmd == "train":
        train(
            data_root=args.data_root,
            csv_map=args.csv_map,
            sr=args.sr,
            win_sec=args.win_sec,
            hop_sec=args.hop_sec,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            lambda_phys=args.lambda_phys,
            lambda_stft=args.lambda_stft,
            alpha_prime_val=args.alpha_prime,
            beta_prime_val=args.beta_prime,
            learn_alpha=args.learn_alpha,
            learn_beta=args.learn_beta,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            num_workers=args.num_workers,
            log_dir=args.log_dir,
            save_every=args.save_every,
            warmup_epochs=args.warmup_epochs
        )
    elif args.cmd == "eval":
        eval_and_export(args.checkpoint, args.data_root, csv_map=args.csv_map, out_dir=args.out_dir, sr=args.sr, win_sec=args.win_sec, hop_sec=args.hop_sec, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
