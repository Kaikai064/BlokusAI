"""Render a board State to an RGB image (NumPy), and map pixels to cells.

No Pillow dependency: produces an (H, W, 3) uint8 array, which Gradio's
gr.Image accepts directly. Kept free of Gradio so it can be unit-tested.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

from blokus_core.rules import BOARD_SIZE, START_CELLS

# Colors (R, G, B)
EMPTY = (238, 238, 238)
GRID = (170, 170, 170)
P0 = (60, 120, 220)        # human / player 0 = blue
P1 = (230, 110, 60)        # AI / player 1 = orange
START = (205, 205, 205)    # the two start cells, when empty
HIGHLIGHT = (120, 210, 130)  # legal placement reference cells


def render_board(state, cell_px: int = 30,
                 highlight: Optional[Iterable[Tuple[int, int]]] = None) -> np.ndarray:
    n = BOARD_SIZE
    side = n * cell_px
    img = np.empty((side, side, 3), dtype=np.uint8)
    occ0, occ1 = state.occ[0], state.occ[1]
    hl = set(highlight or ())
    starts = set(START_CELLS)
    for r in range(n):
        for c in range(n):
            idx = r * n + c
            if (occ0 >> idx) & 1:
                color = P0
            elif (occ1 >> idx) & 1:
                color = P1
            elif (r, c) in hl:
                color = HIGHLIGHT
            elif (r, c) in starts:
                color = START
            else:
                color = EMPTY
            img[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px] = color
    # grid lines
    img[::cell_px, :, :] = GRID
    img[:, ::cell_px, :] = GRID
    img[-1, :, :] = GRID
    img[:, -1, :] = GRID
    return img


def cell_from_pixel(x: int, y: int, cell_px: int = 30) -> Tuple[int, int]:
    """Map a click at pixel (x, y) to a board (row, col), clamped on-board."""
    c = min(BOARD_SIZE - 1, max(0, int(x) // cell_px))
    r = min(BOARD_SIZE - 1, max(0, int(y) // cell_px))
    return r, c


DOT = (47, 158, 94)


def _draw_dot(img: np.ndarray, idx: int, cell_px: int, color=DOT) -> None:
    r, c = divmod(idx, BOARD_SIZE)
    cy, cx = r * cell_px + cell_px // 2, c * cell_px + cell_px // 2
    rad = max(2, cell_px // 7)
    y0, y1 = max(0, cy - rad), min(img.shape[0], cy + rad + 1)
    x0, x1 = max(0, cx - rad), min(img.shape[1], cx + rad + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad
    img[y0:y1, x0:x1][mask] = color


def render_play(state, cell_px: int = 34, legal_refs=None,
                footprint=None, foot_ok: bool = True) -> np.ndarray:
    """Board image with green legal-start dots (and an optional footprint overlay)."""
    img = render_board(state, cell_px)
    if footprint:
        color = (120, 210, 130) if foot_ok else (210, 120, 120)
        for (r, c) in footprint:
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                img[r * cell_px + 2:(r + 1) * cell_px - 1,
                    c * cell_px + 2:(c + 1) * cell_px - 1] = color
    for idx in (legal_refs or ()):
        _draw_dot(img, idx, cell_px)
    return img


def render_piece(variant_cells, color=(60, 120, 220), cell_px: int = 15,
                 pad: int = 3) -> np.ndarray:
    """Small thumbnail image of one oriented piece (for the piece tray)."""
    maxr = max(r for r, _ in variant_cells)
    maxc = max(c for _, c in variant_cells)
    h, w = (maxr + 1) * cell_px + 2 * pad, (maxc + 1) * cell_px + 2 * pad
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for (r, c) in variant_cells:
        y, x = pad + r * cell_px, pad + c * cell_px
        img[y:y + cell_px - 1, x:x + cell_px - 1] = color
    return img
