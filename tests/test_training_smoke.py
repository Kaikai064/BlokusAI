"""End-to-end pipeline smoke test (skipped if torch is unavailable).

Proves the training machinery is sound: a single batch can be overfit (so the
net + loss + optimizer learn), and the full loop runs, checkpoints, and resumes.
The real training run happens on Colab; this just guards correctness.
"""
from __future__ import annotations

import os
import random

import pytest

torch = pytest.importorskip("torch")

from blokus_core.net import BlokusNet, NetEvaluator
from training import loop, selfplay
from training import train as train_mod
from training.config import smoke_config
from training.replay import ReplayBuffer


def test_overfit_single_batch_decreases_loss():
    cfg = smoke_config("unused")
    net = BlokusNet(cfg.channels, cfg.blocks)
    ev = NetEvaluator(net, "cpu")
    rng = random.Random(0)

    samples = []
    while len(samples) < cfg.batch_size:
        s, _ = selfplay.play_game(ev, cfg, rng)
        samples.extend(s)
    batch = train_mod.make_batch(samples[:cfg.batch_size], "cpu")

    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    first = train_mod.train_step(net, opt, batch)["loss"]
    last = first
    for _ in range(40):
        last = train_mod.train_step(net, opt, batch)["loss"]
    assert last < first, (first, last)


def test_loop_runs_checkpoints_and_resumes(tmp_path):
    cfg = smoke_config(str(tmp_path / "ck"))
    loop.run(cfg, resume=False, log=lambda *a, **k: None)

    assert os.path.exists(os.path.join(cfg.ckpt_dir, "manifest.json"))
    assert os.path.exists(os.path.join(cfg.ckpt_dir, "buffer.pkl"))

    # Fresh objects resume from the checkpoint.
    net = BlokusNet(cfg.channels, cfg.blocks)
    opt = torch.optim.AdamW(net.parameters())
    buf = ReplayBuffer(cfg.buffer_size)
    rng = random.Random(0)
    restored_iter = loop.load_checkpoint(cfg, net, opt, buf, rng)
    assert restored_iter == cfg.num_iters
    assert len(buf) > 0
