"""Gradio app: play Blokus Duo against the trained AI.

You are blue (player 0) and move first; the AI is orange. Pick a piece and an
orientation, and the legal placements light up green -- click one to place the
piece (the highlighted square is its top-left corner). The AI then replies.

Weights are loaded from $BLOKUS_WEIGHTS (default ``models/blokus_net.pt``); if
absent, an untrained (weak) network is used so the UI still runs.

Run locally:   python app.py        (or:  SHARE=1 python app.py  for a link)
Requires:      pip install -r requirements.txt
"""
from __future__ import annotations

import os
import random

import gradio as gr

from blokus_core import board
from blokus_core.board import State
from blokus_core.pieces import PIECE_NAMES, PIECE_VARIANTS
from . import render, game_io

WEIGHTS = os.environ.get("BLOKUS_WEIGHTS", "models/blokus_net.pt")
CELL_PX = 30
HUMAN, AI = 0, 1

_EVAL = game_io.make_evaluator(WEIGHTS if os.path.exists(WEIGHTS) else None)
_HAS_WEIGHTS = os.path.exists(WEIGHTS)


def _remaining(state):
    return [PIECE_NAMES[k] for k in range(21) if not state.used[HUMAN][k]]


def _status(state, msg):
    s0, s1 = board.score(state, 0), board.score(state, 1)
    tag = "" if _HAS_WEIGHTS else "  ·  ⚠ untrained model"
    return f"**{msg}**  ·  You (blue) {s0} : {s1} AI (orange){tag}"


def _final(state):
    s0, s1 = board.score(state, 0), board.score(state, 1)
    who = "🎉 You win!" if s0 > s1 else ("AI wins." if s1 > s0 else "It's a draw.")
    return _status(state, f"Game over — {who}")


def _advance_until_human(state, n_sims):
    """Let the AI (and any forced human passes) play until it's the human's
    turn with a legal move, or the game ends."""
    while not board.is_terminal(state):
        if state.current == HUMAN:
            if board.legal_actions(state):
                break
            state = board.apply_action(state, board.PASS)     # human stuck -> pass
        elif board.legal_actions(state):
            _, state = game_io.ai_move(state, _EVAL, n_sims=int(n_sims),
                                       rng=random.Random())
        else:
            state = board.apply_action(state, board.PASS)      # AI stuck -> pass
    return state


def new_game(n_sims):
    state = State.initial()
    return ({"state": state},
            render.render_board(state, CELL_PX),
            gr.update(choices=_remaining(state), value=None),
            gr.update(choices=[], value=None),
            _status(state, "New game — you are blue and move first."))


def pick_piece(gs, piece_name):
    state = gs["state"]
    if not piece_name:
        return render.render_board(state, CELL_PX), gr.update(choices=[], value=None), \
            _status(state, "Pick a piece.")
    n = len(PIECE_VARIANTS[PIECE_NAMES.index(piece_name)])
    return (render.render_board(state, CELL_PX),
            gr.update(choices=[str(i) for i in range(n)], value="0"),
            _status(state, f"{piece_name}: choose an orientation."))


def pick_orientation(gs, piece_name, orient):
    state = gs["state"]
    if not piece_name or orient in (None, ""):
        return render.render_board(state, CELL_PX), _status(state, "Pick a piece and orientation.")
    vid = game_io.variant_id(piece_name, orient)
    refs = game_io.legal_ref_cells(state, vid)
    return (render.render_board(state, CELL_PX, highlight=refs),
            _status(state, f"{len(refs)} legal spots highlighted — click one."))


def click_board(gs, piece_name, orient, n_sims, evt: gr.SelectData):
    state = gs["state"]
    if board.is_terminal(state):
        return gs, render.render_board(state, CELL_PX), gr.update(), _final(state)
    if not piece_name or orient in (None, ""):
        return gs, render.render_board(state, CELL_PX), gr.update(), \
            _status(state, "Pick a piece + orientation first.")
    vid = game_io.variant_id(piece_name, orient)
    cell = render.cell_from_pixel(evt.index[0], evt.index[1], CELL_PX)
    aid = game_io.resolve_human_move(state, vid, cell)
    if aid is None:
        img = render.render_board(state, CELL_PX, highlight=game_io.legal_ref_cells(state, vid))
        return gs, img, gr.update(), _status(state, "Not legal there — click a highlighted square.")
    state = _advance_until_human(board.apply_action(state, aid), n_sims)
    gs["state"] = state
    msg = _final(state) if board.is_terminal(state) else _status(state, "Your move.")
    return (gs, render.render_board(state, CELL_PX),
            gr.update(choices=_remaining(state), value=None), msg)


def pass_move(gs, n_sims):
    state = gs["state"]
    if board.is_terminal(state):
        return gs, render.render_board(state, CELL_PX), gr.update(), _final(state)
    state = _advance_until_human(board.apply_action(state, board.PASS), n_sims)
    gs["state"] = state
    msg = _final(state) if board.is_terminal(state) else _status(state, "Your move.")
    return (gs, render.render_board(state, CELL_PX),
            gr.update(choices=_remaining(state), value=None), msg)


def build_app():
    with gr.Blocks(title="Blokus Duo AI") as demo:
        gr.Markdown("# Blokus Duo — play the AI\n"
                    "You are **blue** and move first. Pick a piece + orientation, "
                    "then click a **green** highlighted square to place it.")
        gs = gr.State({})
        with gr.Row():
            with gr.Column(scale=3):
                board_img = gr.Image(label="Board", interactive=False,
                                     show_download_button=False, height=14 * CELL_PX)
            with gr.Column(scale=2):
                piece = gr.Dropdown(label="Piece", choices=[])
                orient = gr.Dropdown(label="Orientation", choices=[])
                n_sims = gr.Slider(50, 800, value=200, step=50,
                                   label="AI strength (MCTS simulations)")
                with gr.Row():
                    pass_btn = gr.Button("Pass")
                    new_btn = gr.Button("New game", variant="primary")
                status = gr.Markdown()

        new_btn.click(new_game, [n_sims], [gs, board_img, piece, orient, status])
        piece.change(pick_piece, [gs, piece], [board_img, orient, status])
        orient.change(pick_orientation, [gs, piece, orient], [board_img, status])
        board_img.select(click_board, [gs, piece, orient, n_sims],
                         [gs, board_img, piece, status])
        pass_btn.click(pass_move, [gs, n_sims], [gs, board_img, piece, status])
        demo.load(new_game, [n_sims], [gs, board_img, piece, orient, status])
    return demo


if __name__ == "__main__":
    build_app().launch(share=os.environ.get("SHARE", "0") == "1")
