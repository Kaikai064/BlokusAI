"""The AlphaZero iteration loop with Colab-friendly checkpoint/resume.

Each iteration: generate self-play games -> push to the replay buffer ->
train -> checkpoint everything to ``cfg.ckpt_dir`` (which on Colab points at
Google Drive). Checkpoints are written atomically with the manifest last, so a
crash mid-write never corrupts a resume. ``run(resume=True)`` picks up from the
latest good checkpoint.
"""
from __future__ import annotations

import json
import os
import random
import time

import torch

from blokus_core.net import BlokusNet, NetEvaluator
from .replay import ReplayBuffer
from .selfplay import play_game
from .train import make_batch, train_step


def save_checkpoint(cfg, iteration, net, opt, buffer, rng) -> None:
    d = cfg.ckpt_dir
    os.makedirs(d, exist_ok=True)
    ckpt_name = f"ckpt_{iteration:04d}.pt"
    ckpt_path = os.path.join(d, ckpt_name)
    tmp = ckpt_path + ".tmp"
    torch.save({"iteration": iteration, "net": net.state_dict(),
                "opt": opt.state_dict(), "rng": rng.getstate(),
                "config": vars(cfg)}, tmp)
    os.replace(tmp, ckpt_path)
    if cfg.save_buffer:
        buffer.save(os.path.join(d, "buffer.pkl"))
    manifest = {"latest": ckpt_name, "iteration": iteration,
                "buffer": "buffer.pkl" if cfg.save_buffer else None}
    mtmp = os.path.join(d, "manifest.json.tmp")
    with open(mtmp, "w") as f:
        json.dump(manifest, f)
    os.replace(mtmp, os.path.join(d, "manifest.json"))   # written last = atomic


def load_checkpoint(cfg, net, opt, buffer, rng) -> int:
    mpath = os.path.join(cfg.ckpt_dir, "manifest.json")
    if not os.path.exists(mpath):
        return 0
    with open(mpath) as f:
        man = json.load(f)
    ck = torch.load(os.path.join(cfg.ckpt_dir, man["latest"]), map_location=cfg.device)
    net.load_state_dict(ck["net"])
    opt.load_state_dict(ck["opt"])
    rng.setstate(ck["rng"])
    if man.get("buffer"):
        bp = os.path.join(cfg.ckpt_dir, man["buffer"])
        if os.path.exists(bp):
            buffer.load(bp)
    buffer.recap(cfg.buffer_size)   # drop stale data if buffer_size was lowered
    return ck["iteration"]


def evaluate(cfg, net, rng):
    """Win rate (score rate) of the current net vs Random and vs Blocking."""
    from eval.arena import play_match
    from eval.baselines import RandomPlayer, GreedyBlockingPlayer
    from blokus_core.mcts import MCTSPlayer

    ev = NetEvaluator(net, cfg.device)
    pairs = max(1, cfg.eval_games // 2)
    make_net = lambda: MCTSPlayer(ev, n_sims=cfg.eval_sims, temperature=0.0,
                                  rng=random.Random(rng.randrange(2 ** 31)))
    vs_random = play_match(
        make_net, lambda: RandomPlayer(random.Random(rng.randrange(2 ** 31))), pairs)
    vs_block = play_match(
        make_net, lambda: GreedyBlockingPlayer(random.Random(rng.randrange(2 ** 31))), pairs)
    return vs_random["score_rate"], vs_block["score_rate"]


def run(cfg, resume: bool = True, log=print):
    device = cfg.device
    net = BlokusNet(cfg.channels, cfg.blocks).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_size)
    rng = random.Random(cfg.seed)

    start = load_checkpoint(cfg, net, opt, buffer, rng) if resume else 0
    if start:
        log(f"Resumed from iteration {start} (buffer {len(buffer)}).")

    for it in range(start, cfg.num_iters):
        t0 = time.perf_counter()
        n_samples = 0
        if cfg.batched_selfplay:
            from .batched_selfplay import NetBatchEvaluator, generate_games
            bev = NetBatchEvaluator(net, device)
            remaining = cfg.games_per_iter
            while remaining > 0:
                b = min(cfg.selfplay_batch, remaining)
                samples = generate_games(bev, cfg, rng, b)
                buffer.extend(samples)
                n_samples += len(samples)
                remaining -= b
        else:
            ev = NetEvaluator(net, device)
            for _ in range(cfg.games_per_iter):
                samples, _ = play_game(ev, cfg, rng)
                buffer.extend(samples)
                n_samples += len(samples)

        losses = []
        if len(buffer) >= cfg.min_buffer:
            for _ in range(cfg.train_steps_per_iter):
                batch = make_batch(buffer.sample(cfg.batch_size, rng), device)
                losses.append(train_step(net, opt, batch, cfg.value_loss_weight))

        save_checkpoint(cfg, it + 1, net, opt, buffer, rng)
        dt = time.perf_counter() - t0

        def avg(k):
            return sum(d[k] for d in losses) / len(losses) if losses else float("nan")

        msg = (f"iter {it + 1}/{cfg.num_iters}  samples {n_samples}  "
               f"buffer {len(buffer)}  loss {avg('loss'):.3f} "
               f"(p {avg('policy'):.3f} v {avg('value'):.3f})  {dt:.1f}s")
        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            sr_rand, sr_block = evaluate(cfg, net, rng)
            msg += f"  | vs random {sr_rand*100:.0f}%  vs blocking {sr_block*100:.0f}%"
        log(msg)
    return net
