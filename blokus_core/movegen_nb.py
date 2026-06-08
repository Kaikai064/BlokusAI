"""Numba-accelerated legal move generation.

Mirrors the tested pure-Python ``movegen`` exactly, but represents the board as
a ``uint16[14]`` row array (bit c of row r = cell (r, c)) so the hot nested
loops can be JIT-compiled by Numba. If Numba is not installed, ``@njit`` becomes
a no-op and the identical code runs as plain Python+NumPy (correct, just slow) --
so the logic can be validated anywhere (tests/test_movegen_nb.py), while the
~50-200x speedup kicks in on Colab where Numba is available.

Enable in the engine with ``blokus_core.board.set_numba(True)`` AFTER the fuzz
test passes. The first move (own == 0) falls back to the pure-Python generator
(it happens once per game and needs the start-cell special case).
"""
from __future__ import annotations

import numpy as np

from .rules import BOARD_SIZE, START_INDICES
from .pieces import (
    VARIANTS, VARIANT_PIECE, NUM_VARIANTS, NUM_PIECES, NUM_CELLS,
)
from . import movegen

try:                                   # pragma: no cover - environment dependent
    from numba import njit as _njit

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _njit()(args[0])
        return _njit(*args, **kwargs)

    HAVE_NUMBA = True
except Exception:                      # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(fn):
            return fn
        return deco


# --- Numba-friendly variant tables (computed once) ---------------------------
_MAXLEN = max(len(v) for v in VARIANTS)                       # 5
# int32 (not int8) so the action-id arithmetic below stays in a wide integer
# type under NumPy's NEP50 rules when running as plain Python (no Numba).
VARIANT_CELLS = np.full((NUM_VARIANTS, _MAXLEN, 2), 0, dtype=np.int32)
VARIANT_LEN = np.zeros(NUM_VARIANTS, dtype=np.int32)
VARIANT_PIECE_ARR = np.asarray(VARIANT_PIECE, dtype=np.int32)
for _v, _cells in enumerate(VARIANTS):
    VARIANT_LEN[_v] = len(_cells)
    for _j, (_r, _c) in enumerate(_cells):
        VARIANT_CELLS[_v, _j, 0] = _r
        VARIANT_CELLS[_v, _j, 1] = _c

_MASK14 = 0x3FFF                                              # low 14 bits


def _to_rows(occ_int: int) -> np.ndarray:
    rows = np.empty(BOARD_SIZE, dtype=np.uint16)
    for r in range(BOARD_SIZE):
        rows[r] = (occ_int >> (r * BOARD_SIZE)) & _MASK14
    return rows


@njit(cache=True)
def _gen_core(own, opp, used, vcells, vlen, vpiece, buf):
    N = 14
    occ_all = np.zeros(N, dtype=np.uint16)
    forbidden = np.zeros(N, dtype=np.uint16)
    anchors = np.zeros(N, dtype=np.uint16)
    for r in range(N):
        occ_all[r] = own[r] | opp[r]
    # forbidden = occ_all | ortho(own)
    for r in range(N):
        row = own[r]
        horiz = ((row << 1) | (row >> 1)) & 0x3FFF
        forbidden[r] = forbidden[r] | horiz
        if r > 0:
            forbidden[r - 1] = forbidden[r - 1] | row
        if r < N - 1:
            forbidden[r + 1] = forbidden[r + 1] | row
    for r in range(N):
        forbidden[r] = forbidden[r] | occ_all[r]
    # anchors = diag(own) & ~forbidden
    for r in range(N):
        row = own[r]
        d = ((row << 1) | (row >> 1)) & 0x3FFF
        if r > 0:
            anchors[r - 1] = anchors[r - 1] | d
        if r < N - 1:
            anchors[r + 1] = anchors[r + 1] | d
    for r in range(N):
        anchors[r] = anchors[r] & (0x3FFF ^ forbidden[r])

    seen = np.zeros(91 * 196, dtype=np.uint8)
    cnt = 0
    V = vcells.shape[0]
    for ar in range(N):
        arow = anchors[ar]
        if arow == 0:
            continue
        for ac in range(N):
            if ((arow >> ac) & 1) == 0:
                continue
            for v in range(V):
                if used[vpiece[v]] != 0:
                    continue
                L = vlen[v]
                for oi in range(L):
                    ref_r = ar - vcells[v, oi, 0]
                    ref_c = ac - vcells[v, oi, 1]
                    if ref_r < 0 or ref_c < 0:
                        continue
                    ok = True
                    for j in range(L):
                        R = ref_r + vcells[v, j, 0]
                        C = ref_c + vcells[v, j, 1]
                        if R >= N or C >= N:
                            ok = False
                            break
                        if ((forbidden[R] >> C) & 1) != 0:
                            ok = False
                            break
                    if ok:
                        aid = v * 196 + (ref_r * N + ref_c)
                        if seen[aid] == 0:
                            seen[aid] = 1
                            buf[cnt] = aid
                            cnt += 1
    return cnt


def _legal_ids(own_int, opp_int, start_index, used_list):
    if own_int == 0:                          # first move -> pure-Python path
        return [int(a) for a in movegen.legal_actions(
            own_int, opp_int, start_index, used_list)]
    own = _to_rows(own_int)
    opp = _to_rows(opp_int)
    used = np.asarray([1 if u else 0 for u in used_list], dtype=np.int8)
    buf = np.empty(8192, dtype=np.int64)
    n = _gen_core(own, opp, used, VARIANT_CELLS, VARIANT_LEN, VARIANT_PIECE_ARR, buf)
    return [int(buf[i]) for i in range(n)]


def legal_actions_state(state):
    p = state.current
    return _legal_ids(state.occ[p], state.occ[1 - p], START_INDICES[p], state.used[p])


def has_legal_state(state, player):
    return len(_legal_ids(
        state.occ[player], state.occ[1 - player],
        START_INDICES[player], state.used[player])) > 0


def warmup():
    """Trigger JIT compilation (and validate the call path) once."""
    own = _to_rows(1 << (4 * BOARD_SIZE + 4))
    opp = _to_rows(0)
    used = np.zeros(NUM_PIECES, dtype=np.int8)
    buf = np.empty(8192, dtype=np.int64)
    _gen_core(own, opp, used, VARIANT_CELLS, VARIANT_LEN, VARIANT_PIECE_ARR, buf)
