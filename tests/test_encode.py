"""Encoding: input planes and the policy legal-action mask."""
from __future__ import annotations

import random

import numpy as np

from blokus_core import board, encode
from blokus_core.board import State
from blokus_core.rules import cell_index
from blokus_core.encode import NUM_PLANES


def test_plane_shape_and_dtype():
    planes = encode.encode_state(State.initial())
    assert planes.shape == (NUM_PLANES, 14, 14)
    assert planes.dtype == np.float32


def test_initial_planes_content():
    planes = encode.encode_state(State.initial())
    # Board empty -> no own/opp stones, everything empty.
    assert planes[0].sum() == 0          # own occupied
    assert planes[1].sum() == 0          # opp occupied
    assert planes[2].sum() == 196        # empty
    # First-move anchor is the player-0 start cell (4,4).
    assert planes[3].sum() == 1
    assert planes[3][4, 4] == 1.0
    # All 21 pieces available for both players.
    assert planes[5:5 + 21].sum() == 21 * 196
    assert planes[5 + 21:5 + 42].sum() == 21 * 196
    # Move count is 0.
    assert planes[47].sum() == 0


def test_perspective_swaps_after_move():
    s = State.initial()
    a = board.legal_actions(s)[0]
    s2 = board.apply_action(s, a)
    # Now it's player 1 to move: player 0's stones appear on the OPPONENT plane.
    planes = encode.encode_state(s2)
    assert planes[0].sum() == 0                       # mover (P1) has nothing yet
    assert planes[1].sum() == board.squares_placed(s2, 0)  # opponent = P0's piece
    # P1's first-move anchor is the (9,9) start cell.
    assert planes[3][9, 9] == 1.0


def test_legal_mask_matches_legal_actions():
    rng = random.Random(7)
    s = State.initial()
    for _ in range(8):
        acts = board.legal_actions(s)
        if not acts:
            break
        mask = encode.legal_mask(s)
        assert set(np.nonzero(mask)[0].tolist()) == set(acts)
        assert mask.sum() == len(acts)
        s = board.apply_action(s, rng.choice(acts))


def test_pieces_remaining_plane_updates():
    s = State.initial()
    a = board.legal_actions(s)[0]
    s2 = board.apply_action(s, a)
    # From P0's perspective again (skip P1): build a state where it's P0's turn.
    s3 = board.apply_action(s2, board.PASS)  # P1 passes -> back to P0
    planes = encode.encode_state(s3)
    # P0 used exactly one piece -> 20 of the 21 own-remaining planes are full.
    own_remaining_full = sum(1 for k in range(21) if planes[5 + k].sum() == 196)
    assert own_remaining_full == 20
