"""Encode a game State into neural-network input planes, and build the policy
legal-action mask.

All planes are from the **side-to-move's perspective**: plane group "own"
always refers to the player whose turn it is, so the network always reasons as
"the player about to move". The value head therefore predicts the result from
the mover's point of view.

Plane layout (NUM_PLANES = 48), each 14x14 float32:
    0           own occupied
    1           opponent occupied
    2           empty
    3           own anchors (legal corner-touch cells; the start cell on move 1)
    4           own forbidden (occupied or edge-adjacent to own)
    5 .. 25     own pieces remaining   (21 constant planes, 1.0 if available)
    26 .. 46    opponent pieces remaining (21 constant planes)
    47          move number, normalized to [0, 1]
"""
from __future__ import annotations

import numpy as np

from .rules import BOARD_SIZE, NUM_CELLS, FULL_MASK, START_INDICES
from .pieces import NUM_PIECES, ACTION_SPACE
from . import movegen, board

NUM_PLANES = 3 + 2 + NUM_PIECES + NUM_PIECES + 1   # 48
_MOVE_NORM = 40.0  # games run ~30-40 plies; normalizer for the move-count plane


def bitboard_to_plane(bb: int) -> np.ndarray:
    """Convert a 196-bit bitboard to a 14x14 float32 plane."""
    arr = np.zeros(NUM_CELLS, dtype=np.float32)
    while bb:
        low = bb & (-bb)
        arr[low.bit_length() - 1] = 1.0
        bb ^= low
    return arr.reshape(BOARD_SIZE, BOARD_SIZE)


def encode_state(state: board.State) -> np.ndarray:
    """State -> (NUM_PLANES, 14, 14) float32 tensor, from the mover's view."""
    p = state.current
    own, opp = state.occ[p], state.occ[1 - p]
    occ_all = own | opp
    empty = (~occ_all) & FULL_MASK
    anchors = movegen.anchor_mask(own, opp, START_INDICES[p])
    forbidden = occ_all | movegen.ortho(own)

    planes = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    planes[0] = bitboard_to_plane(own)
    planes[1] = bitboard_to_plane(opp)
    planes[2] = bitboard_to_plane(empty)
    planes[3] = bitboard_to_plane(anchors)
    planes[4] = bitboard_to_plane(forbidden)

    own_base = 5
    opp_base = 5 + NUM_PIECES
    for k in range(NUM_PIECES):
        if not state.used[p][k]:
            planes[own_base + k] = 1.0
        if not state.used[1 - p][k]:
            planes[opp_base + k] = 1.0

    planes[5 + 2 * NUM_PIECES] = min(state.num_moves / _MOVE_NORM, 1.0)
    return planes


def legal_mask(state: board.State) -> np.ndarray:
    """Boolean array of length ACTION_SPACE; True at legal action ids."""
    mask = np.zeros(ACTION_SPACE, dtype=bool)
    for a in board.legal_actions(state):
        mask[a] = True
    return mask


def legal_mask_from_actions(actions) -> np.ndarray:
    """Build a legal mask from a precomputed list of action ids (avoids a second
    move generation when the caller already has the legal actions)."""
    mask = np.zeros(ACTION_SPACE, dtype=bool)
    for a in actions:
        mask[a] = True
    return mask
