"""Generate one self-play game with MCTS, recording training targets.

The policy target stored at each position is the MCTS visit-count distribution
(temperature 1). Move *selection* uses a temperature schedule: sample
proportional to visits for the first ``temp_moves`` plies (exploration), then
play the most-visited move. Root Dirichlet noise is enabled for exploration.
Torch-free (the neural net is reached only via the evaluator).
"""
from __future__ import annotations

import numpy as np

from blokus_core import board
from blokus_core.board import State
from blokus_core.mcts import MCTS


def snapshot(s: State):
    """Compact, picklable State fields sufficient to rebuild input planes."""
    return (s.occ[0], s.occ[1], tuple(s.used[0]), tuple(s.used[1]),
            s.current, s.num_moves)


def play_game(evaluator, cfg, rng):
    """Play one self-play game. Returns (samples, final_state)."""
    s = State.initial()
    mcts = MCTS(evaluator, n_sims=cfg.n_sims, c_puct=cfg.c_puct,
                dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_eps=cfg.dirichlet_eps,
                rng=rng)
    records = []
    consecutive_pass = 0
    ply = 0
    while consecutive_pass < 2 and ply < cfg.max_plies:
        if not board.legal_actions(s):
            s = board.apply_action(s, board.PASS)
            consecutive_pass += 1
            ply += 1
            continue
        root = mcts.run(s, add_noise=True)
        counts = root.N.astype(np.float64)
        if counts.sum() == 0:                       # degenerate (n_sims == 0)
            pi = np.ones(len(root.legal)) / len(root.legal)
            a_idx = 0
        else:
            pi = counts / counts.sum()
            if ply < cfg.temp_moves:
                a_idx = int(mcts.np_rng.choice(len(counts), p=pi))
            else:
                a_idx = int(np.argmax(counts))
        records.append((snapshot(s),
                        np.asarray(root.legal, dtype=np.int32),
                        pi.astype(np.float32),
                        s.current))
        s = board.apply_action(s, root.legal[a_idx])
        consecutive_pass = 0
        ply += 1

    z = board.outcome(s)                             # +1 if player 0 won
    samples = [(snap, acts, pi, float(z if player == 0 else -z))
               for (snap, acts, pi, player) in records]
    return samples, s
