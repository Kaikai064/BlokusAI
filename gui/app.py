"""Gradio app: play Blokus Duo against the trained neural-net AI.

Friendly UX: click a piece in your tray, Rotate/Flip to orient it, and green
dots mark every legal placement -- click one to place. The AI (the trained
network + MCTS) then replies; raise the "AI strength" slider for more sims
(stronger but slower on CPU).

You are BLUE (player 0) and move first; the AI is ORANGE. Weights load from
$BLOKUS_WEIGHTS (default models/blokus_net.pt); without them an untrained
(weak) net is used so the UI still runs.

Run locally:   python app.py        (SHARE=1 python app.py  for a public link)
"""
from __future__ import annotations

import os
import random

import gradio as gr

from blokus_core import board
from blokus_core.board import State
from blokus_core.pieces import (
    PIECE_NAMES, PIECE_VARIANTS, VARIANTS, NUM_PIECES, ROT_OF, FLIP_OF, decode,
)
from . import render, game_io

WEIGHTS = os.environ.get("BLOKUS_WEIGHTS", "models/blokus_net.pt")
CELL = 34
HUMAN, AI = 0, 1
PIECE_COLOR = render.P0  # human tray thumbnails match the board's player-0 color

_EVAL = game_io.make_evaluator(WEIGHTS if os.path.exists(WEIGHTS) else None)
_HAS_WEIGHTS = os.path.exists(WEIGHTS)


# --- view helpers ---------------------------------------------------------
def _remaining(s):
    return [pid for pid in range(NUM_PIECES) if not s.used[HUMAN][pid]]


def _legal_refs(s, variant):
    if variant is None or s.current != HUMAN:
        return []
    return [rc for a in board.legal_actions(s) for (v, rc) in [decode(a)] if v == variant]


def _gallery(s):
    return [(render.render_piece(VARIANTS[PIECE_VARIANTS[pid][0]], PIECE_COLOR), PIECE_NAMES[pid])
            for pid in _remaining(s)]


def _board_img(gs):
    s = gs["s"]
    return render.render_play(s, CELL, legal_refs=_legal_refs(s, gs["variant"]))


def _status(gs):
    s = gs["s"]
    s0, s1 = board.score(s, 0), board.score(s, 1)
    left = sum(1 for k in range(NUM_PIECES) if not s.used[HUMAN][k])
    tag = "" if _HAS_WEIGHTS else "  ·  ⚠ untrained net"
    if board.is_terminal(s):
        o = board.outcome(s)
        head = "🎉 You win!" if o > 0 else ("Computer wins" if o < 0 else "Draw")
        return f"### {head}\nFinal — You **{s0}** · Computer **{s1}**"
    legal = len(board.legal_actions(s))
    hint = gs.get("msg") or ("Pick a piece →" if gs["variant"] is None
                             else "Click a green dot to place. Rotate/Flip to reorient.")
    return (f"**Your turn (blue)** · You {s0} : {s1} AI · {left} pieces left · "
            f"{legal} legal moves{tag}\n\n{hint}")


def _advance(s, n_sims):
    """Let the AI (and forced human passes) play until it's the human's turn."""
    while not board.is_terminal(s):
        if s.current == HUMAN:
            if board.legal_actions(s):
                break
            s = board.apply_action(s, board.PASS)
        elif board.legal_actions(s):
            _, s = game_io.ai_move(s, _EVAL, n_sims=int(n_sims), rng=random.Random())
        else:
            s = board.apply_action(s, board.PASS)
    return s


# --- event handlers -------------------------------------------------------
def new_game(_n_sims):
    s = State.initial(HUMAN)
    gs = {"s": s, "pid": None, "variant": None, "msg": ""}
    return gs, _board_img(gs), _gallery(s), _status(gs)


def select_piece(gs, evt: gr.SelectData):
    s = gs["s"]
    rem = _remaining(s)
    if s.current == HUMAN and not board.is_terminal(s) and evt.index is not None and evt.index < len(rem):
        pid = rem[evt.index]
        gs["pid"], gs["variant"], gs["msg"] = pid, PIECE_VARIANTS[pid][0], ""
    return gs, _board_img(gs), _status(gs)


def reorient(gs, table):
    if gs["variant"] is not None:
        gs["variant"] = table[gs["variant"]]
    return gs, _board_img(gs), _status(gs)


def skip(gs):
    gs["pid"], gs["variant"], gs["msg"] = None, None, ""
    return gs, _board_img(gs), _status(gs)


def click_board(gs, n_sims, evt: gr.SelectData):
    s = gs["s"]
    if not board.is_terminal(s) and s.current == HUMAN:
        if gs["variant"] is None:
            gs["msg"] = "Pick a piece first."
        else:
            r, c = render.cell_from_pixel(evt.index[0], evt.index[1], CELL)
            aid = game_io.resolve_human_move(s, gs["variant"], (r, c))
            if aid is None:
                gs["msg"] = "Not legal there — click a green dot."
            else:
                s = _advance(board.apply_action(s, aid), n_sims)
                gs["s"], gs["pid"], gs["variant"], gs["msg"] = s, None, None, ""
    return gs, _board_img(gs), _gallery(gs["s"]), _status(gs)


def pass_move(gs, n_sims):
    s = gs["s"]
    if not board.is_terminal(s) and s.current == HUMAN:
        s = _advance(board.apply_action(s, board.PASS), n_sims)
        gs["s"], gs["pid"], gs["variant"], gs["msg"] = s, None, None, ""
    return gs, _board_img(gs), _gallery(gs["s"]), _status(gs)


def build_app():
    with gr.Blocks(title="Blokus Duo AI") as demo:
        gr.Markdown("# Blokus Duo — play the AI\n"
                    "You are **blue** and move first. Pick a piece, **Rotate/Flip**, "
                    "then click a **green dot** to place it.")
        gs = gr.State({})
        with gr.Row():
            with gr.Column(scale=3):
                board_img = gr.Image(label="Board", interactive=False,
                                     show_download_button=False, height=14 * CELL)
                status = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    rot_b = gr.Button("⟳ Rotate")
                    flip_b = gr.Button("⇄ Flip")
                    skip_b = gr.Button("Skip")
                    pass_b = gr.Button("Pass")
                n_sims = gr.Slider(50, 1200, value=300, step=50, label="AI strength (MCTS sims)")
                new_b = gr.Button("New game", variant="primary")
                gallery = gr.Gallery(label="Your pieces — click to select", columns=4,
                                     height=380, allow_preview=False)

        new_b.click(new_game, [n_sims], [gs, board_img, gallery, status])
        gallery.select(select_piece, [gs], [gs, board_img, status])
        rot_b.click(lambda gs: reorient(gs, ROT_OF), [gs], [gs, board_img, status])
        flip_b.click(lambda gs: reorient(gs, FLIP_OF), [gs], [gs, board_img, status])
        skip_b.click(skip, [gs], [gs, board_img, status])
        pass_b.click(pass_move, [gs, n_sims], [gs, board_img, gallery, status])
        board_img.select(click_board, [gs, n_sims], [gs, board_img, gallery, status])
        demo.load(new_game, [n_sims], [gs, board_img, gallery, status])
    return demo


if __name__ == "__main__":
    build_app().launch(share=os.environ.get("SHARE", "0") == "1")
