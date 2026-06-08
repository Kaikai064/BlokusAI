"""All training hyperparameters in one dataclass.

Defaults target a real Colab GPU run; ``smoke_config`` shrinks everything for a
fast local end-to-end pipeline test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # --- network ---
    channels: int = 96
    blocks: int = 8
    # --- MCTS / self-play ---
    n_sims: int = 128
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temp: float = 1.0
    temp_moves: int = 14          # sample for the first N plies, then play argmax
    games_per_iter: int = 50
    max_plies: int = 120
    selfplay_batch: int = 32       # games run concurrently for batched NN eval
    batched_selfplay: bool = True
    # --- training ---
    batch_size: int = 256
    train_steps_per_iter: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    # --- replay buffer ---
    buffer_size: int = 200_000
    min_buffer: int = 2_000       # don't train until the buffer has this many
    # --- loop / checkpointing ---
    num_iters: int = 100
    seed: int = 0
    device: str = field(default_factory=_default_device)
    ckpt_dir: str = "checkpoints"
    save_buffer: bool = True
    # --- evaluation ---
    eval_every: int = 5           # 0 disables periodic eval vs baselines
    eval_games: int = 20
    eval_sims: int = 128


def smoke_config(ckpt_dir: str) -> Config:
    """Tiny configuration for a fast local pipeline smoke test."""
    return Config(
        channels=16, blocks=2,
        n_sims=6, games_per_iter=2, temp_moves=6, selfplay_batch=2,
        batch_size=32, train_steps_per_iter=20,
        buffer_size=5_000, min_buffer=16,
        num_iters=1, device="cpu", ckpt_dir=ckpt_dir,
        save_buffer=True, eval_every=0,
    )
