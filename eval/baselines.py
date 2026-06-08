"""Baseline opponents, from weak to strong. These form the strength ladder the
trained AlphaZero network is measured against (Tier A is defined relative to
them). Every player exposes ``select(state) -> action_id`` (or board.PASS).
"""
from __future__ import annotations

import random

from blokus_core import board, movegen
from blokus_core.rules import START_INDICES
from blokus_core.pieces import PIECE_SIZE, VARIANT_PIECE, NUM_CELLS


def _piece_size(action: int) -> int:
    return PIECE_SIZE[VARIANT_PIECE[action // NUM_CELLS]]


def _popcount(x: int) -> int:
    return bin(x).count("1")


class RandomPlayer:
    """Uniformly random legal move (ladder rung B0)."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def select(self, state):
        acts = board.legal_actions(state)
        return self.rng.choice(acts) if acts else board.PASS


class GreedyPlayer:
    """Always place the largest available piece (most squares) -- rung B1."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def select(self, state):
        acts = board.legal_actions(state)
        if not acts:
            return board.PASS
        best = max(_piece_size(a) for a in acts)
        return self.rng.choice([a for a in acts if _piece_size(a) == best])


class GreedyBlockingPlayer:
    """Maximize squares placed and own mobility while denying the opponent's.

    Score of a move = w_size * (piece squares)
                      + w_own  * (own anchors after the move)
                      - w_opp  * (opponent anchors after the move)

    Blocking the opponent's expansion is often stronger than greedily grabbing
    area, so this is the most informative cheap baseline (rung B3).
    """

    def __init__(self, rng=None, w_size=2.0, w_own=1.0, w_opp=1.0):
        self.rng = rng or random.Random()
        self.w_size = w_size
        self.w_own = w_own
        self.w_opp = w_opp

    def select(self, state):
        acts = board.legal_actions(state)
        if not acts:
            return board.PASS
        p = state.current
        best_score = None
        top = []
        for a in acts:
            ns = board.apply_action(state, a)
            own, opp = ns.occ[p], ns.occ[1 - p]
            own_anc = _popcount(movegen.anchor_mask(own, opp, START_INDICES[p]))
            opp_anc = _popcount(movegen.anchor_mask(opp, own, START_INDICES[1 - p]))
            score = self.w_size * _piece_size(a) + self.w_own * own_anc - self.w_opp * opp_anc
            if best_score is None or score > best_score:
                best_score, top = score, [a]
            elif score == best_score:
                top.append(a)
        return self.rng.choice(top)
