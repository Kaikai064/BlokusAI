"""Batched self-play search correctness + game generation.

The headline test: with a uniform evaluator and a single game, the batched
search must reproduce the trusted recursive MCTS visit counts exactly.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from blokus_core import mcts
from blokus_core.board import State
from training import batched_selfplay as bsp
from training.config import smoke_config


def test_batched_search_matches_recursive_mcts_uniform():
    s = State.initial()
    recursive = mcts.MCTS(mcts.UniformEvaluator(), n_sims=64, c_puct=1.5,
                          rng=random.Random(0)).run(s, add_noise=False)
    rec_counts = dict(zip(recursive.legal, recursive.N.astype(int)))

    roots = bsp.search_batch([s.copy()], bsp.UniformBatchEvaluator(),
                             n_sims=64, c_puct=1.5, add_noise=False)
    bat_counts = dict(zip(roots[0].legal, roots[0].N.astype(int)))

    assert rec_counts == bat_counts


def test_batched_search_runs_for_many_games():
    states = [State.initial() for _ in range(8)]
    roots = bsp.search_batch(states, bsp.UniformBatchEvaluator(), n_sims=32, c_puct=1.5)
    assert len(roots) == 8
    for r in roots:
        assert int(r.N.sum()) == 32


def test_generate_games_uniform_valid_samples():
    cfg = smoke_config("unused")
    samples = bsp.generate_games(bsp.UniformBatchEvaluator(), cfg, random.Random(0),
                                 num_games=3)
    assert len(samples) > 0
    for _snap, acts, pi, z in samples:
        assert len(acts) == len(pi)
        assert abs(float(pi.sum()) - 1.0) < 1e-5
        assert z in (-1.0, 0.0, 1.0)


def test_generate_games_with_net():
    pytest.importorskip("torch")
    from blokus_core.net import BlokusNet
    cfg = smoke_config("unused")
    ev = bsp.NetBatchEvaluator(BlokusNet(channels=16, blocks=2), "cpu")
    samples = bsp.generate_games(ev, cfg, random.Random(0), num_games=2)
    assert len(samples) > 0
