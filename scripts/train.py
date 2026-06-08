"""Entry point for AlphaZero training. Designed to be called from a Colab cell
or locally. Resumes automatically from the latest checkpoint in --ckpt-dir.

Examples:
    python scripts/train.py --ckpt-dir /content/drive/MyDrive/blokus_ckpt
    python scripts/train.py --num-iters 200 --games-per-iter 80
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import Config  # noqa: E402
from training import loop  # noqa: E402


def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Train the Blokus Duo AlphaZero net")
    ap.add_argument("--ckpt-dir", default=cfg.ckpt_dir)
    ap.add_argument("--num-iters", type=int, default=cfg.num_iters)
    ap.add_argument("--games-per-iter", type=int, default=cfg.games_per_iter)
    ap.add_argument("--n-sims", type=int, default=cfg.n_sims)
    ap.add_argument("--channels", type=int, default=cfg.channels)
    ap.add_argument("--blocks", type=int, default=cfg.blocks)
    ap.add_argument("--device", default=cfg.device)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg.ckpt_dir = args.ckpt_dir
    cfg.num_iters = args.num_iters
    cfg.games_per_iter = args.games_per_iter
    cfg.n_sims = args.n_sims
    cfg.channels = args.channels
    cfg.blocks = args.blocks
    cfg.device = args.device

    print(f"Training on {cfg.device}; checkpoints -> {cfg.ckpt_dir}")
    loop.run(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()
