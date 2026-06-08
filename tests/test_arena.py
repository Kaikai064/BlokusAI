"""Baselines + arena: stronger heuristics should beat weaker ones."""
from __future__ import annotations

import random

from eval.baselines import RandomPlayer, GreedyPlayer, GreedyBlockingPlayer
from eval.arena import play_match, play_game


def _seeded(factory, master):
    return lambda: factory(random.Random(master.randrange(2 ** 31)))


def test_greedy_beats_random():
    m = random.Random(0)
    res = play_match(_seeded(GreedyPlayer, m), _seeded(RandomPlayer, m), num_pairs=15)
    assert res["score_rate"] > 0.8, res


def test_blocking_beats_random():
    m = random.Random(1)
    res = play_match(_seeded(GreedyBlockingPlayer, m), _seeded(RandomPlayer, m), num_pairs=15)
    assert res["score_rate"] > 0.85, res


def test_paired_match_is_balanced_and_complete():
    m = random.Random(2)
    res = play_match(_seeded(RandomPlayer, m), _seeded(RandomPlayer, m), num_pairs=10)
    assert res["games"] == 20
    assert res["a_wins"] + res["draws"] + res["b_wins"] == 20
    assert 0.0 <= res["score_rate"] <= 1.0


def test_play_game_reaches_terminal():
    o, s = play_game(RandomPlayer(random.Random(3)), RandomPlayer(random.Random(4)))
    from blokus_core import board
    assert board.is_terminal(s)
    assert o in (-1, 0, 1)
