"""Round-robin between the baseline players, reported as color-swapped matches.

Run:  python scripts/play_baselines.py [num_pairs]
"""
from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.arena import play_match  # noqa: E402
from eval.baselines import (  # noqa: E402
    RandomPlayer, GreedyPlayer, GreedyBlockingPlayer,
)


def seeded(factory, master):
    return lambda: factory(random.Random(master.randrange(2 ** 31)))


def main(num_pairs: int = 25) -> None:
    master = random.Random(2024)
    contests = [
        ("Greedy",   GreedyPlayer,         "Random",   RandomPlayer),
        ("Blocking", GreedyBlockingPlayer, "Random",   RandomPlayer),
        ("Blocking", GreedyBlockingPlayer, "Greedy",   GreedyPlayer),
    ]
    print(f"Color-swapped matches, {num_pairs} pairs ({2 * num_pairs} games) each:\n")
    for a_name, a_cls, b_name, b_cls in contests:
        t0 = time.perf_counter()
        res = play_match(seeded(a_cls, master), seeded(b_cls, master), num_pairs=num_pairs)
        dt = time.perf_counter() - t0
        print(f"  {a_name:>8} vs {b_name:<8}  "
              f"score {res['score_rate']*100:5.1f}%  "
              f"(W{res['a_wins']} D{res['draws']} L{res['b_wins']})  [{dt:.1f}s]")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    main(n)
