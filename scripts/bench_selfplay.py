"""Compare unbatched vs batched self-play throughput.

Batching fuses many games' MCTS leaf evaluations into single network forward
passes. On CPU the gain is moderate (amortized per-call overhead); on a GPU it
is much larger because per-call launch latency dominates.

Run:  python scripts/bench_selfplay.py [num_games]
"""
from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus_core.net import BlokusNet, NetEvaluator  # noqa: E402
from training.config import Config  # noqa: E402
from training.selfplay import play_game  # noqa: E402
from training.batched_selfplay import NetBatchEvaluator, generate_games  # noqa: E402


def main(num_games: int = 6) -> None:
    net = BlokusNet(channels=16, blocks=2)
    cfg = Config(n_sims=16, temp_moves=8, device="cpu", max_plies=120)

    ev = NetEvaluator(net, "cpu")
    t0 = time.perf_counter()
    n = 0
    for _ in range(num_games):
        samples, _ = play_game(ev, cfg, random.Random(0))
        n += len(samples)
    t_unb = time.perf_counter() - t0

    bev = NetBatchEvaluator(net, "cpu")
    t0 = time.perf_counter()
    generate_games(bev, cfg, random.Random(0), num_games)
    t_bat = time.perf_counter() - t0

    print(f"net=16ch/2blk  n_sims={cfg.n_sims}  games={num_games}  (CPU)")
    print(f"  unbatched: {t_unb:5.1f}s  ({num_games / t_unb:.2f} games/s)")
    print(f"  batched:   {t_bat:5.1f}s  ({num_games / t_bat:.2f} games/s)")
    print(f"  speedup:   {t_unb / t_bat:.2f}x  (much larger on GPU)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
