"""Blokus Duo engine core.

Shared, dependency-light game logic (rules, pieces, board, move generation)
that is imported by BOTH the training pipeline and the GUI, so there is a
single source of truth for the rules.
"""

from . import rules, pieces, board, movegen, encode

__all__ = ["rules", "pieces", "board", "movegen", "encode"]
