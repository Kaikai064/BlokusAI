"""Move-generation correctness -- the highest-value tests in the project.

The fast anchor-based generator is fuzzed against the independent brute-force
reference over many random games; they must agree at every position.
"""
from __future__ import annotations

import random

import pytest

from blokus_core import board, movegen
from blokus_core.board import State, PASS
from blokus_core.rules import START_INDICES, cell_index
from blokus_core.pieces import encode_action, NUM_PIECES


# --- Targeted neighbour-mask tests (catch wraparound bugs) -------------------
def test_ortho_center_cell():
    s = 1 << cell_index(4, 4)
    expected = {cell_index(3, 4), cell_index(5, 4), cell_index(4, 3), cell_index(4, 5)}
    assert set(movegen.bits(movegen.ortho(s))) == expected


def test_diag_center_cell():
    s = 1 << cell_index(4, 4)
    expected = {cell_index(3, 3), cell_index(3, 5), cell_index(5, 3), cell_index(5, 5)}
    assert set(movegen.bits(movegen.diag(s))) == expected


def test_neighbours_do_not_wrap_at_edges():
    # Top-left corner (0,0): neighbours stay on-board, no wrap to row above/left.
    s = 1 << cell_index(0, 0)
    assert set(movegen.bits(movegen.ortho(s))) == {cell_index(0, 1), cell_index(1, 0)}
    assert set(movegen.bits(movegen.diag(s))) == {cell_index(1, 1)}
    # Top-right corner (0,13).
    s = 1 << cell_index(0, 13)
    assert set(movegen.bits(movegen.ortho(s))) == {cell_index(0, 12), cell_index(1, 13)}
    assert set(movegen.bits(movegen.diag(s))) == {cell_index(1, 12)}
    # Bottom-right corner (13,13).
    s = 1 << cell_index(13, 13)
    assert set(movegen.bits(movegen.ortho(s))) == {cell_index(13, 12), cell_index(12, 13)}
    assert set(movegen.bits(movegen.diag(s))) == {cell_index(12, 12)}


# --- First-move rules --------------------------------------------------------
def test_first_move_must_cover_start_cell():
    s = State.initial()
    acts = board.legal_actions(s)
    assert len(acts) > 0
    start = (4, 4)
    for a in acts:
        from blokus_core.pieces import placement_cells
        assert start in placement_cells(a), a


def test_first_move_matches_bruteforce():
    s = State.initial()
    fast = set(board.legal_actions(s))
    brute = movegen.legal_actions_bruteforce(0, 0, START_INDICES[0], s.used[0])
    assert fast == brute


# --- Corner/edge legality on a hand-built position ---------------------------
def test_corner_touch_legal_edge_touch_illegal():
    own = 1 << cell_index(4, 4)            # one own stone at the centre
    opp = 0
    used = [False] * NUM_PIECES            # all pieces still available
    legal = set(movegen.legal_actions(own, opp, START_INDICES[0], used))

    # Monomino (variant 0) on the four diagonal cells -> legal (corner touch).
    for (r, c) in [(3, 3), (3, 5), (5, 3), (5, 5)]:
        assert encode_action(0, cell_index(r, c)) in legal, (r, c)
    # Monomino on the four orthogonal cells -> illegal (edge touch).
    for (r, c) in [(3, 4), (5, 4), (4, 3), (4, 5)]:
        assert encode_action(0, cell_index(r, c)) not in legal, (r, c)
    # Monomino on the occupied cell itself -> illegal.
    assert encode_action(0, cell_index(4, 4)) not in legal


def test_can_touch_opponent_freely():
    # Own stone at (4,4); opponent wall directly right at (4,5).
    own = 1 << cell_index(4, 4)
    opp = 1 << cell_index(4, 5)
    used = [False] * NUM_PIECES
    legal = set(movegen.legal_actions(own, opp, START_INDICES[0], used))
    # Placing the monomino at (3,5): diagonal to own (legal corner-touch) AND
    # edge-adjacent to the opponent -- which is allowed.
    assert encode_action(0, cell_index(3, 5)) in legal
    # The opponent's own cell is still occupied -> cannot place there.
    assert encode_action(0, cell_index(4, 5)) not in legal


# --- The fuzz test: fast generator == brute force over random games ----------
@pytest.mark.parametrize("seed", range(12))
def test_fuzz_fast_matches_bruteforce_over_random_games(seed):
    rng = random.Random(seed)
    s = State.initial(first_player=seed % 2)
    consecutive_pass = 0
    plies = 0
    while consecutive_pass < 2 and plies < 100:
        p = s.current
        fast = set(movegen.legal_actions(
            s.occ[p], s.occ[1 - p], START_INDICES[p], s.used[p]))
        brute = movegen.legal_actions_bruteforce(
            s.occ[p], s.occ[1 - p], START_INDICES[p], s.used[p])
        assert fast == brute, f"seed={seed} ply={plies} player={p}"
        # has_any_legal must agree with the generators.
        assert board.has_legal(s, p) == (len(fast) > 0)
        if fast:
            s = board.apply_action(s, rng.choice(sorted(fast)))
            consecutive_pass = 0
        else:
            s = board.apply_action(s, PASS)
            consecutive_pass += 1
        plies += 1

    assert board.is_terminal(s)
    assert board.outcome(s) in (-1, 0, 1)


def test_random_game_runs_to_terminal():
    rng = random.Random(123)
    s = board.play_random(State.initial(), rng)
    assert board.is_terminal(s)
    # Every placed piece is a real piece; scores are within achievable bounds.
    for p in (0, 1):
        assert 0 <= board.squares_placed(s, p) <= 89
