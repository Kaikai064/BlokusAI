"""Verify the piece set and orientation tables are exactly correct."""
from __future__ import annotations

import pytest

from blokus_core import pieces
from blokus_core.pieces import (
    NUM_PIECES, NUM_VARIANTS, ACTION_SPACE, PIECE_NAMES, PIECE_SIZE,
    PIECE_VARIANTS, VARIANTS, free_canonical, normalize,
    encode_action, decode_action, placement_cells,
)
from blokus_core.rules import NUM_CELLS

# Known number of fixed orientations per free polyomino.
EXPECTED_ORIENTATIONS = {
    "I1": 1, "I2": 2,
    "I3": 2, "V3": 4,
    "I4": 2, "O4": 1, "T4": 4, "S4": 4, "L4": 8,
    "F5": 8, "I5": 2, "L5": 8, "N5": 8, "P5": 8, "T5": 4,
    "U5": 4, "V5": 4, "W5": 4, "X5": 1, "Y5": 8, "Z5": 4,
}


def test_piece_counts():
    assert NUM_PIECES == 21
    assert NUM_VARIANTS == 91
    assert ACTION_SPACE == 91 * NUM_CELLS == 17836


def test_per_piece_orientation_counts():
    for pid, name in enumerate(PIECE_NAMES):
        assert len(PIECE_VARIANTS[pid]) == EXPECTED_ORIENTATIONS[name], name
    assert sum(EXPECTED_ORIENTATIONS.values()) == 91


def test_piece_sizes():
    expected = [1, 2, 3, 3, 4, 4, 4, 4, 4] + [5] * 12
    assert PIECE_SIZE == expected


def test_pieces_are_distinct_free_polyominoes():
    canon = {free_canonical(VARIANTS[PIECE_VARIANTS[pid][0]]) for pid in range(NUM_PIECES)}
    assert len(canon) == NUM_PIECES  # all 21 shapes are different


# --- Independent enumeration of free polyominoes -----------------------------
def _grow(polys):
    nxt = set()
    for poly in polys:
        for (r, c) in poly:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cell = (r + dr, c + dc)
                if cell not in poly:
                    nxt.add(frozenset(poly | {cell}))
    return nxt


def _enumerate_free(n):
    polys = {frozenset({(0, 0)})}
    for _ in range(n - 1):
        polys = _grow(polys)
    return {free_canonical(tuple(p)) for p in polys}


# Free polyomino counts: 1, 1, 2, 5, 12 for sizes 1..5.
@pytest.mark.parametrize("size,count", [(1, 1), (2, 1), (3, 2), (4, 5), (5, 12)])
def test_base_shapes_match_all_free_polyominoes(size, count):
    enumerated = _enumerate_free(size)
    assert len(enumerated) == count
    mine = {
        free_canonical(VARIANTS[PIECE_VARIANTS[pid][0]])
        for pid in range(NUM_PIECES) if PIECE_SIZE[pid] == size
    }
    assert mine == enumerated


def test_action_encode_decode_roundtrip():
    for variant_id in (0, 1, 45, 90):
        for ref_cell in (0, 60, 195):
            a = encode_action(variant_id, ref_cell)
            assert decode_action(a) == (variant_id, ref_cell)


def test_placement_cells_for_monomino_at_start():
    # variant 0 is the monomino; ref cell (4,4)=60 -> single cell (4,4).
    a = encode_action(0, 60)
    assert placement_cells(a) == [(4, 4)]


def test_variants_are_normalized_and_unique_within_piece():
    for pid in range(NUM_PIECES):
        seen = set()
        for v in PIECE_VARIANTS[pid]:
            shape = VARIANTS[v]
            assert shape == normalize(shape)          # already normalized
            assert min(r for r, _ in shape) == 0
            assert min(c for _, c in shape) == 0
            assert shape not in seen                   # unique orientation
            seen.add(shape)


def test_orientation_transition_maps():
    from blokus_core.pieces import ROT_OF, FLIP_OF, NUM_VARIANTS
    for v in range(NUM_VARIANTS):
        assert ROT_OF[ROT_OF[ROT_OF[ROT_OF[v]]]] == v   # 4 rotations = identity
        assert FLIP_OF[FLIP_OF[v]] == v                 # 2 flips = identity
