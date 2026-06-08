"""Core rules and constants for Blokus Duo (2-player, 14x14).

Verified rule facts (do not change without re-checking sources):
  * Board is 14x14 = 196 cells.
  * Two fixed start cells near the centre (NOT the corners):
        Player 0 must cover (4, 4); Player 1 must cover (9, 9)  [0-indexed].
  * First piece must cover the player's start cell.
  * Every later piece must touch one of your own pieces corner-to-corner
    (diagonally) and must NOT share an edge with any of your own pieces.
  * You may touch opponent pieces in any way.
  * Scoring (higher wins): +1 per square placed; +15 bonus if you place all 21
    pieces; +5 further if your very last placed piece was the monomino.
"""
from __future__ import annotations

from typing import Tuple

BOARD_SIZE = 14
NUM_CELLS = BOARD_SIZE * BOARD_SIZE          # 196
FULL_MASK = (1 << NUM_CELLS) - 1
NUM_PLAYERS = 2

# Zero-indexed start cells: (row, col) for player 0 and player 1.
START_CELLS: Tuple[Tuple[int, int], Tuple[int, int]] = ((4, 4), (9, 9))

# Scoring bonuses.
ALL_PLACED_BONUS = 15
MONOMINO_LAST_BONUS = 5


def cell_index(r: int, c: int) -> int:
    """(row, col) -> flat bit/array index."""
    return r * BOARD_SIZE + c


def cell_coords(idx: int) -> Tuple[int, int]:
    """flat index -> (row, col)."""
    return divmod(idx, BOARD_SIZE)


START_INDICES: Tuple[int, int] = (
    cell_index(*START_CELLS[0]),
    cell_index(*START_CELLS[1]),
)
