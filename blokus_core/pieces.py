"""The 21 Blokus pieces and their distinct fixed orientations ("variants").

A *variant* is one fixed orientation (a rotation and/or reflection) of a piece,
with its cells normalized so the bounding-box origin sits at (0, 0). Across the
21 pieces there are exactly **91** variants.

We define each piece by a single base shape and then generate all 8 dihedral
(D4) images, normalizing and de-duplicating. This makes the orientation set
correct *by construction* -- only the free-polyomino identity of each base shape
matters, not the particular base orientation we happen to write down.

Action encoding
---------------
    action_id = variant_id * NUM_CELLS + ref_cell

where ``ref_cell`` is the flat board index on which the variant's normalized
origin (0, 0) lands (i.e. the top-left of the placement's bounding box). Because
every variant has a distinct normalized cell-set, this is a bijection with
concrete (oriented-shape, position) placements.
"""
from __future__ import annotations

from typing import List, Tuple

from .rules import BOARD_SIZE, NUM_CELLS

Cell = Tuple[int, int]
Shape = Tuple[Cell, ...]

# --- Base shapes for the 21 pieces (standard polyomino names) ----------------
# Index 0 is the monomino; this ordering is relied upon by board.MONOMINO_PIECE_ID.
_BASE_PIECES: List[Tuple[str, Shape]] = [
    # Monomino (1 square)
    ("I1", ((0, 0),)),
    # Domino (2)
    ("I2", ((0, 0), (0, 1))),
    # Trominoes (3)
    ("I3", ((0, 0), (0, 1), (0, 2))),
    ("V3", ((0, 0), (1, 0), (1, 1))),
    # Tetrominoes (4)
    ("I4", ((0, 0), (0, 1), (0, 2), (0, 3))),
    ("O4", ((0, 0), (0, 1), (1, 0), (1, 1))),
    ("T4", ((0, 0), (0, 1), (0, 2), (1, 1))),
    ("S4", ((0, 1), (0, 2), (1, 0), (1, 1))),
    ("L4", ((0, 0), (1, 0), (2, 0), (2, 1))),
    # Pentominoes (5)
    ("F5", ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1))),
    ("I5", ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))),
    ("L5", ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1))),
    ("N5", ((0, 1), (1, 1), (2, 0), (2, 1), (3, 0))),
    ("P5", ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))),
    ("T5", ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))),
    ("U5", ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2))),
    ("V5", ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2))),
    ("W5", ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2))),
    ("X5", ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))),
    ("Y5", ((0, 1), (1, 0), (1, 1), (2, 1), (3, 1))),
    ("Z5", ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2))),
]

NUM_PIECES = len(_BASE_PIECES)            # 21
PIECE_NAMES: Tuple[str, ...] = tuple(name for name, _ in _BASE_PIECES)


# --- Geometry helpers --------------------------------------------------------
def normalize(cells) -> Shape:
    """Translate so min row and min col are 0; return a sorted tuple."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate90(cells) -> Shape:
    """Rotate 90 degrees: (r, c) -> (c, -r)."""
    return tuple((c, -r) for r, c in cells)


def _reflect(cells) -> Shape:
    """Mirror across a vertical axis: (r, c) -> (r, -c)."""
    return tuple((r, -c) for r, c in cells)


def all_orientations(shape: Shape) -> List[Shape]:
    """All distinct normalized D4 images (1..8) of ``shape``."""
    seen = set()
    out: List[Shape] = []
    cur = shape
    for _ in range(4):
        for img in (cur, _reflect(cur)):
            norm = normalize(img)
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        cur = _rotate90(cur)
    return out


def free_canonical(cells) -> Shape:
    """Canonical representative of a *free* polyomino (min over all D4 images).

    Two cell-sets are the same free polyomino iff they share this value.
    """
    best = None
    cur = tuple(cells)
    for _ in range(4):
        for img in (cur, _reflect(cur)):
            norm = normalize(img)
            if best is None or norm < best:
                best = norm
        cur = _rotate90(cur)
    return best  # type: ignore[return-value]


# --- Build the variant tables (computed once at import) ----------------------
def _build_variants():
    variants: List[Shape] = []
    variant_piece: List[int] = []
    piece_variants: List[List[int]] = []
    piece_size: List[int] = []
    for pid, (_name, shape) in enumerate(_BASE_PIECES):
        norm_shape = normalize(shape)
        piece_size.append(len(norm_shape))
        idxs: List[int] = []
        for orient in all_orientations(norm_shape):
            idxs.append(len(variants))
            variants.append(orient)
            variant_piece.append(pid)
        piece_variants.append(idxs)
    return variants, variant_piece, piece_variants, piece_size


VARIANTS, VARIANT_PIECE, PIECE_VARIANTS, PIECE_SIZE = _build_variants()
NUM_VARIANTS = len(VARIANTS)               # 91
ACTION_SPACE = NUM_VARIANTS * NUM_CELLS    # 17836

# Per-variant bounding box extents (max row/col offset), handy for move-gen.
VARIANT_MAX = tuple((max(r for r, _ in v), max(c for _, c in v)) for v in VARIANTS)

# Orientation transition maps for UI rotate/flip buttons: variant -> variant.
_KEY_TO_VARIANT = {v: i for i, v in enumerate(VARIANTS)}
ROT_OF = tuple(_KEY_TO_VARIANT[normalize(_rotate90(v))] for v in VARIANTS)
FLIP_OF = tuple(_KEY_TO_VARIANT[normalize(_reflect(v))] for v in VARIANTS)


# --- Action <-> placement conversions ----------------------------------------
def decode_action(action_id: int) -> Tuple[int, int]:
    """action_id -> (variant_id, ref_cell)."""
    return divmod(action_id, NUM_CELLS)


def encode_action(variant_id: int, ref_cell: int) -> int:
    """(variant_id, ref_cell) -> action_id."""
    return variant_id * NUM_CELLS + ref_cell


def placement_cells(action_id: int) -> List[Cell]:
    """The list of board (row, col) cells occupied by ``action_id``."""
    variant_id, ref_cell = decode_action(action_id)
    rr, cc = divmod(ref_cell, BOARD_SIZE)
    return [(rr + r, cc + c) for (r, c) in VARIANTS[variant_id]]


def piece_of_action(action_id: int) -> int:
    """The piece id (0..20) that ``action_id`` would place."""
    variant_id, _ = decode_action(action_id)
    return VARIANT_PIECE[variant_id]
