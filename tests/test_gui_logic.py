"""GUI logic: board rendering, pixel mapping, and click->move resolution.
(The Gradio app itself isn't imported here; only the tested helpers are.)"""
from __future__ import annotations

import random

import numpy as np
import pytest

from blokus_core import board
from blokus_core.board import State
from blokus_core.pieces import encode_action
from gui import render, game_io


def test_render_board_shape_and_dtype():
    img = render.render_board(State.initial(), cell_px=16)
    assert img.shape == (14 * 16, 14 * 16, 3)
    assert img.dtype == np.uint8


def test_render_shows_placed_stones():
    s = State.initial()
    s = board.apply_action(s, encode_action(0, 4 * 14 + 4))  # blue monomino at (4,4)
    img = render.render_board(s, cell_px=10)
    # center of cell (4,4) should be the player-0 color
    px = img[4 * 10 + 5, 4 * 10 + 5]
    assert tuple(int(v) for v in px) == render.P0


def test_cell_from_pixel_inverse():
    for r in (0, 3, 13):
        for c in (0, 7, 13):
            assert render.cell_from_pixel(c * 30 + 7, r * 30 + 7, 30) == (r, c)


def test_legal_ref_cells_and_resolve_first_move():
    s = State.initial()
    refs = game_io.legal_ref_cells(s, 0)          # monomino variant
    assert (4, 4) in refs                          # must be able to cover start
    assert game_io.resolve_human_move(s, 0, (4, 4)) == encode_action(0, 4 * 14 + 4)
    assert game_io.resolve_human_move(s, 0, (0, 0)) is None


def test_ai_move_advances_state():
    torch = pytest.importorskip("torch")
    from blokus_core.net import BlokusNet, NetEvaluator
    s = State.initial()
    ev = NetEvaluator(BlokusNet(channels=16, blocks=2), "cpu")
    a, ns = game_io.ai_move(s, ev, n_sims=8, rng=random.Random(0))
    assert ns.num_moves == 1
    assert a in board.legal_actions(s)
