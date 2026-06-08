"""Rule constants and scoring."""
from __future__ import annotations

from blokus_core import board
from blokus_core.rules import (
    BOARD_SIZE, NUM_CELLS, START_CELLS, START_INDICES, cell_index,
)
from blokus_core.board import State, MONOMINO_PIECE_ID


def test_board_constants():
    assert BOARD_SIZE == 14
    assert NUM_CELLS == 196


def test_start_cells_are_4_4_and_9_9():
    assert START_CELLS == ((4, 4), (9, 9))
    assert START_INDICES == (cell_index(4, 4), cell_index(9, 9)) == (60, 135)


def test_monomino_is_piece_zero():
    from blokus_core.pieces import PIECE_NAMES, PIECE_SIZE
    assert MONOMINO_PIECE_ID == 0
    assert PIECE_NAMES[0] == "I1"
    assert PIECE_SIZE[0] == 1


def test_score_counts_placed_squares():
    s = State.initial()
    # Player 0 has placed the monomino (1 square) and the I5 pentomino (5).
    s.used[0][0] = True       # I1
    s.used[0][10] = True      # I5
    assert board.squares_placed(s, 0) == 6
    assert board.score(s, 0) == 6        # no all-placed bonus
    assert board.squares_placed(s, 1) == 0


def test_all_placed_bonus_and_monomino_last():
    s = State.initial()
    for k in range(21):
        s.used[0][k] = True
    s.last_piece[0] = 5                   # last piece was not the monomino
    # 89 squares + 15 all-placed bonus.
    assert board.score(s, 0) == 89 + 15
    s.last_piece[0] = MONOMINO_PIECE_ID   # last piece was the monomino
    assert board.score(s, 0) == 89 + 15 + 5


def test_outcome_sign():
    s = State.initial()
    s.used[0][10] = True                  # player 0: 5 squares
    s.used[1][0] = True                   # player 1: 1 square
    assert board.outcome(s) == 1
    s.used[1][9] = True                   # player 1: +5 -> 6 squares > 5
    assert board.outcome(s) == -1
