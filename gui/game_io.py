"""Glue between the GUI and the engine: resolve clicks into legal moves, run the
AI's move, and load trained weights. Gradio-free so it can be unit-tested.
"""
from __future__ import annotations

import random
from typing import Optional, Set, Tuple

from blokus_core import board
from blokus_core.board import State
from blokus_core.pieces import (
    PIECE_NAMES, PIECE_VARIANTS, decode_action, encode_action,
)
from blokus_core.mcts import MCTSPlayer
from blokus_core.rules import BOARD_SIZE


def variant_id(piece_name: str, orientation_index: int) -> int:
    pid = PIECE_NAMES.index(piece_name)
    return PIECE_VARIANTS[pid][int(orientation_index)]


def legal_ref_cells(state: State, variant: int) -> Set[Tuple[int, int]]:
    """Reference cells (bbox top-left) where ``variant`` can legally be placed."""
    out = set()
    for a in board.legal_actions(state):
        v, ref = decode_action(a)
        if v == variant:
            out.add(divmod(ref, BOARD_SIZE))
    return out


def resolve_human_move(state: State, variant: int,
                       clicked_cell: Tuple[int, int]) -> Optional[int]:
    """The action id for placing ``variant`` with its reference at ``clicked_cell``,
    or None if that is not a legal move."""
    r, c = clicked_cell
    aid = encode_action(variant, r * BOARD_SIZE + c)
    return aid if aid in set(board.legal_actions(state)) else None


def ai_move(state: State, evaluator, n_sims: int = 200, rng=None):
    """Pick and apply the AI's move. Returns (action_id, new_state)."""
    if not board.legal_actions(state):
        return board.PASS, board.apply_action(state, board.PASS)
    player = MCTSPlayer(evaluator, n_sims=n_sims, temperature=0.0,
                        rng=rng or random.Random())
    a = player.select(state)
    return a, board.apply_action(state, a)


def load_net(path: str, device: str = "cpu"):
    """Load a BlokusNet from a training checkpoint (.pt with 'net' + 'config')."""
    import torch
    from blokus_core.net import BlokusNet
    ck = torch.load(path, map_location=device)
    cfg = ck.get("config", {})
    net = BlokusNet(cfg.get("channels", 96), cfg.get("blocks", 8))
    net.load_state_dict(ck["net"])
    net.eval()
    return net


def make_evaluator(weights_path: Optional[str] = None, device: Optional[str] = None):
    """NetEvaluator from weights if given, else an untrained (weak) net.

    Device defaults to $BLOKUS_DEVICE, else CUDA if available, else CPU -- so the
    same app is fast on a GPU (Colab) and still works on CPU (HF Spaces free).
    """
    import os
    import torch
    from blokus_core.net import BlokusNet, NetEvaluator
    if device is None:
        device = os.environ.get("BLOKUS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(weights_path, device) if weights_path else BlokusNet()
    return NetEvaluator(net, device)
