"""Characterize and benchmark the Blokus Duo engine.

Reports the opening branching factor, and over many random games: average/max
branching factor, game length, scores, first-player win rate, and raw
move-generation throughput. These numbers inform MCTS design (branching factor),
the self-play temperature schedule (game length), and whether the Numba
acceleration step is urgent (throughput).

Run:  python scripts/bench.py [num_games]
"""
from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus_core import board  # noqa: E402
from blokus_core.board import State, PASS


def main(num_games: int = 200) -> None:
    rng = random.Random(0)

    first_moves = len(board.legal_actions(State.initial()))
    print(f"Legal first moves (player 0): {first_moves}")

    plies_list = []
    branching = []
    scores = []
    outcomes = {-1: 0, 0: 0, 1: 0}
    move_gen_calls = 0

    t0 = time.perf_counter()
    for g in range(num_games):
        s = State.initial(first_player=g % 2)
        consecutive_pass = 0
        plies = 0
        while consecutive_pass < 2 and plies < 100:
            acts = board.legal_actions(s)
            move_gen_calls += 1
            if acts:
                branching.append(len(acts))
                s = board.apply_action(s, rng.choice(acts))
                consecutive_pass = 0
            else:
                s = board.apply_action(s, PASS)
                consecutive_pass += 1
            plies += 1
        plies_list.append(plies)
        scores.append((board.score(s, 0), board.score(s, 1)))
        outcomes[board.outcome(s)] += 1
    elapsed = time.perf_counter() - t0

    avg = lambda xs: sum(xs) / len(xs)
    print(f"\nOver {num_games} random games:")
    print(f"  game length (plies):   avg {avg(plies_list):.1f}  max {max(plies_list)}")
    print(f"  branching factor:      avg {avg(branching):.1f}  max {max(branching)}")
    print(f"  avg score:             P0 {avg([s[0] for s in scores]):.1f}  "
          f"P1 {avg([s[1] for s in scores]):.1f}")
    print(f"  outcomes (P0/draw/P1): {outcomes[1]}/{outcomes[0]}/{outcomes[-1]}")
    print(f"\nThroughput:")
    print(f"  {num_games} games in {elapsed:.2f}s  "
          f"({num_games / elapsed:.1f} games/s, {move_gen_calls / elapsed:.0f} movegen calls/s)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n)
