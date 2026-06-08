"""Validate the Numba move generator against the brute-force reference.

Runs everywhere: without Numba, ``movegen_nb`` executes as plain Python+NumPy
(via the no-op @njit), so this still checks the logic. With Numba installed
(Colab) it additionally exercises the JIT-compiled path.
"""
from __future__ import annotations

import random

import pytest

from blokus_core import board, movegen, movegen_nb
from blokus_core.board import State, PASS
from blokus_core.rules import START_INDICES


@pytest.mark.parametrize("seed", range(3))
def test_numba_matches_bruteforce_over_random_games(seed):
    rng = random.Random(seed)
    s = State.initial(first_player=seed % 2)
    consecutive_pass = 0
    plies = 0
    while consecutive_pass < 2 and plies < 100:
        p = s.current
        nb = set(movegen_nb.legal_actions_state(s))
        brute = movegen.legal_actions_bruteforce(
            s.occ[p], s.occ[1 - p], START_INDICES[p], s.used[p])
        assert nb == brute, f"seed={seed} ply={plies}"
        assert movegen_nb.has_legal_state(s, p) == (len(brute) > 0)
        acts = sorted(brute)
        s = board.apply_action(s, rng.choice(acts)) if acts else board.apply_action(s, PASS)
        consecutive_pass = 0 if acts else consecutive_pass + 1
        plies += 1


def test_set_numba_dispatch_matches_pure_python():
    s = State.initial()
    # advance a couple of plies so own occupancy is non-empty for both players
    s = board.apply_action(s, board.legal_actions(s)[0])
    s = board.apply_action(s, board.legal_actions(s)[0])
    pure = set(board.legal_actions(s))
    board.set_numba(True)
    try:
        nb = set(board.legal_actions(s))
    finally:
        board.set_numba(False)
    assert nb == pure
