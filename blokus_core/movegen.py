"""Legal move generation for Blokus Duo.

Pure-Python bitboard reference implementation. Correctness here is paramount --
every later component (self-play, training, GUI) inherits it -- so this module
ships with both:

  * ``legal_actions``           : the fast, anchor-based generator used in play, and
  * ``legal_actions_bruteforce``: a slow, obviously-correct generator that tries
                                  every placement and checks the rules directly.

The two must always agree (see tests/test_movegen.py, which fuzzes them against
each other over random games). A later step JIT-compiles ``legal_actions`` with
Numba; this reference defines the behaviour it must reproduce.

Board is represented as a 196-bit integer bitboard; bit (r*14 + c) is set when
cell (r, c) is occupied.
"""
from __future__ import annotations

from typing import List, Sequence, Set

from .rules import BOARD_SIZE as N, NUM_CELLS, FULL_MASK
from .pieces import VARIANTS, VARIANT_MAX, PIECE_VARIANTS, NUM_PIECES

# --- Column masks for wraparound-safe neighbour shifts -----------------------
_COL0 = 0
_COL_LAST = 0
for _r in range(N):
    _COL0 |= 1 << (_r * N + 0)
    _COL_LAST |= 1 << (_r * N + (N - 1))
NOT_COL0 = FULL_MASK & ~_COL0
NOT_COL_LAST = FULL_MASK & ~_COL_LAST


def ortho(s: int) -> int:
    """Set of cells orthogonally (edge-) adjacent to the cells in ``s``."""
    up = s >> N
    down = (s << N) & FULL_MASK
    left = (s & NOT_COL0) >> 1
    right = ((s & NOT_COL_LAST) << 1) & FULL_MASK
    return (up | down | left | right) & FULL_MASK


def diag(s: int) -> int:
    """Set of cells diagonally (corner-) adjacent to the cells in ``s``."""
    ul = (s & NOT_COL0) >> (N + 1)
    ur = (s & NOT_COL_LAST) >> (N - 1)
    dl = ((s & NOT_COL0) << (N - 1)) & FULL_MASK
    dr = ((s & NOT_COL_LAST) << (N + 1)) & FULL_MASK
    return (ul | ur | dl | dr) & FULL_MASK


def bits(mask: int) -> List[int]:
    """Indices of set bits in ``mask`` (ascending)."""
    out: List[int] = []
    while mask:
        low = mask & (-mask)
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def _available_variants(used_pieces: Sequence[bool]) -> List[int]:
    out: List[int] = []
    for pid in range(NUM_PIECES):
        if not used_pieces[pid]:
            out.extend(PIECE_VARIANTS[pid])
    return out


def anchor_mask(own: int, opp: int, start_index: int) -> int:
    """Bitboard of cells where a new piece may anchor (corner-touch points).

    On the first move (``own == 0``) the only anchor is the start cell, if free.
    """
    occ_all = own | opp
    if own == 0:
        if (occ_all >> start_index) & 1:
            return 0
        return 1 << start_index
    forbidden = occ_all | ortho(own)
    return diag(own) & ~forbidden & FULL_MASK


def legal_actions(own: int, opp: int, start_index: int,
                  used_pieces: Sequence[bool]) -> List[int]:
    """Legal action ids for the player whose stones are ``own``.

    Parameters
    ----------
    own, opp     : bitboards of this player's and the opponent's stones.
    start_index  : this player's fixed start-cell flat index.
    used_pieces  : length-21 booleans; True means the piece is already placed.
    """
    forbidden = (own | opp) | ortho(own)    # cannot place on these cells
    anchor_indices = bits(anchor_mask(own, opp, start_index))
    if not anchor_indices:
        return []
    variant_ids = _available_variants(used_pieces)
    results: Set[int] = set()

    for a in anchor_indices:
        a_r, a_c = divmod(a, N)
        for v in variant_ids:
            offsets = VARIANTS[v]
            # Try landing each cell of the variant onto the anchor.
            for (o_r, o_c) in offsets:
                ref_r = a_r - o_r
                ref_c = a_c - o_c
                if ref_r < 0 or ref_c < 0:
                    continue
                placement = 0
                ok = True
                for (q_r, q_c) in offsets:
                    R = ref_r + q_r
                    C = ref_c + q_c
                    if R >= N or C >= N:
                        ok = False
                        break
                    placement |= 1 << (R * N + C)
                if not ok or (placement & forbidden):
                    continue
                # Corner-touch is guaranteed: the anchor cell is in the
                # placement and is diagonally adjacent to ``own`` (or is the
                # start cell on the first move).
                results.add(v * NUM_CELLS + (ref_r * N + ref_c))
    return list(results)


def has_any_legal(own: int, opp: int, start_index: int,
                  used_pieces: Sequence[bool]) -> bool:
    """True if the player has at least one legal move (early-exit)."""
    forbidden = (own | opp) | ortho(own)
    anchor_indices = bits(anchor_mask(own, opp, start_index))
    if not anchor_indices:
        return False
    variant_ids = _available_variants(used_pieces)
    for a in anchor_indices:
        a_r, a_c = divmod(a, N)
        for v in variant_ids:
            offsets = VARIANTS[v]
            for (o_r, o_c) in offsets:
                ref_r = a_r - o_r
                ref_c = a_c - o_c
                if ref_r < 0 or ref_c < 0:
                    continue
                placement = 0
                ok = True
                for (q_r, q_c) in offsets:
                    R = ref_r + q_r
                    C = ref_c + q_c
                    if R >= N or C >= N:
                        ok = False
                        break
                    placement |= 1 << (R * N + C)
                if ok and not (placement & forbidden):
                    return True
    return False


def legal_actions_bruteforce(own: int, opp: int, start_index: int,
                             used_pieces: Sequence[bool]) -> Set[int]:
    """Reference generator: try every placement, check the rules from scratch.

    Independent of ``legal_actions`` -- used only to validate it in tests.
    """
    occ_all = own | opp
    own_ortho = ortho(own)
    own_diag = diag(own)
    first = (own == 0)
    results: Set[int] = set()
    for pid in range(NUM_PIECES):
        if used_pieces[pid]:
            continue
        for v in PIECE_VARIANTS[pid]:
            offsets = VARIANTS[v]
            max_r, max_c = VARIANT_MAX[v]
            for ref_r in range(N - max_r):
                for ref_c in range(N - max_c):
                    placement = 0
                    for (q_r, q_c) in offsets:
                        placement |= 1 << ((ref_r + q_r) * N + (ref_c + q_c))
                    if placement & occ_all:
                        continue
                    if first:
                        if (placement >> start_index) & 1:
                            results.add(v * NUM_CELLS + ref_r * N + ref_c)
                    else:
                        if placement & own_ortho:
                            continue          # illegal: edge-touches own
                        if not (placement & own_diag):
                            continue          # illegal: no corner-touch with own
                        results.add(v * NUM_CELLS + ref_r * N + ref_c)
    return results
