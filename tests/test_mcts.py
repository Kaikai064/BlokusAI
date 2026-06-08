"""MCTS (torch-free, using the uniform evaluator)."""
from __future__ import annotations

import random

import numpy as np

from blokus_core import board, mcts
from blokus_core.board import State


def test_policy_is_distribution_over_legal_moves():
    m = mcts.MCTS(mcts.UniformEvaluator(), n_sims=32, rng=random.Random(0))
    legal, probs = m.policy(State.initial(), temperature=1.0)
    assert len(legal) == len(probs)
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    assert set(legal).issubset(set(board.legal_actions(State.initial())))


def test_visit_counts_accumulate():
    m = mcts.MCTS(mcts.UniformEvaluator(), n_sims=50, rng=random.Random(1))
    root = m.run(State.initial())
    assert int(root.N.sum()) == 50


def test_temperature_zero_is_argmax():
    m = mcts.MCTS(mcts.UniformEvaluator(), n_sims=40, rng=random.Random(2))
    legal, probs = m.policy(State.initial(), temperature=0.0)
    assert probs.max() == 1.0
    assert int((probs > 0).sum()) == 1


def test_mcts_player_finishes_game():
    from eval.arena import play_game
    from eval.baselines import RandomPlayer
    p0 = mcts.MCTSPlayer(mcts.UniformEvaluator(), n_sims=16, temperature=0.0,
                         rng=random.Random(0))
    p1 = RandomPlayer(random.Random(1))
    o, s = play_game(p0, p1)
    assert board.is_terminal(s)
    assert o in (-1, 0, 1)
