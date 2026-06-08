"""Mutable-ish game state, move application, terminal detection and scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .rules import (
    BOARD_SIZE, START_INDICES, ALL_PLACED_BONUS, MONOMINO_LAST_BONUS,
)
from .pieces import (
    NUM_PIECES, PIECE_SIZE, VARIANT_PIECE, VARIANTS, decode_action,
)
from . import movegen

PASS = -1
MONOMINO_PIECE_ID = 0  # "I1" is index 0 in pieces._BASE_PIECES

# Optional Numba-accelerated move generation. Off by default (the pure-Python
# path is the tested source of truth); enable with set_numba(True) once the
# movegen_nb fuzz test passes in the target environment.
_NB = None


def set_numba(enabled: bool = True) -> None:
    global _NB
    if enabled:
        from . import movegen_nb
        movegen_nb.warmup()
        _NB = movegen_nb
    else:
        _NB = None


@dataclass
class State:
    """A Blokus Duo position.

    occ[p]        : bitboard of player p's stones
    used[p][k]    : True if player p has placed piece k
    last_piece[p] : piece id of player p's most recently placed piece (-1 = none)
    current       : player to move (0 or 1)
    finished[p]   : True once player p has passed (can no longer move)
    num_moves     : plies played so far
    """
    occ: List[int]
    used: List[List[bool]]
    last_piece: List[int]
    current: int
    finished: List[bool]
    num_moves: int

    @staticmethod
    def initial(first_player: int = 0) -> "State":
        return State(
            occ=[0, 0],
            used=[[False] * NUM_PIECES, [False] * NUM_PIECES],
            last_piece=[-1, -1],
            current=first_player,
            finished=[False, False],
            num_moves=0,
        )

    def copy(self) -> "State":
        return State(
            occ=list(self.occ),
            used=[list(self.used[0]), list(self.used[1])],
            last_piece=list(self.last_piece),
            current=self.current,
            finished=list(self.finished),
            num_moves=self.num_moves,
        )


def placement_mask(action_id: int) -> int:
    """Bitboard of the cells that ``action_id`` would occupy."""
    variant_id, ref_cell = decode_action(action_id)
    rr, cc = divmod(ref_cell, BOARD_SIZE)
    mask = 0
    for (r, c) in VARIANTS[variant_id]:
        mask |= 1 << ((rr + r) * BOARD_SIZE + (cc + c))
    return mask


def legal_actions(state: State) -> List[int]:
    """Legal action ids for the player to move (empty list means must pass)."""
    if _NB is not None:
        return _NB.legal_actions_state(state)
    p = state.current
    return movegen.legal_actions(
        state.occ[p], state.occ[1 - p], START_INDICES[p], state.used[p]
    )


def has_legal(state: State, player: int) -> bool:
    if _NB is not None:
        return _NB.has_legal_state(state, player)
    return movegen.has_any_legal(
        state.occ[player], state.occ[1 - player],
        START_INDICES[player], state.used[player],
    )


def apply_action(state: State, action_id: int) -> State:
    """Return a new state with ``action_id`` (or PASS) applied for the mover."""
    s = state.copy()
    p = s.current
    if action_id == PASS:
        s.finished[p] = True
    else:
        variant_id, _ = decode_action(action_id)
        pid = VARIANT_PIECE[variant_id]
        s.occ[p] |= placement_mask(action_id)
        s.used[p][pid] = True
        s.last_piece[p] = pid
    s.num_moves += 1
    s.current = 1 - p
    return s


def is_terminal(state: State) -> bool:
    """Game ends once neither player can place another piece."""
    return (not has_legal(state, 0)) and (not has_legal(state, 1))


# --- Scoring -----------------------------------------------------------------
def squares_placed(state: State, player: int) -> int:
    return sum(PIECE_SIZE[k] for k in range(NUM_PIECES) if state.used[player][k])


def all_placed(state: State, player: int) -> bool:
    return all(state.used[player])


def score(state: State, player: int) -> int:
    """Blokus score for ``player`` (higher is better)."""
    total = squares_placed(state, player)
    if all_placed(state, player):
        total += ALL_PLACED_BONUS
        if state.last_piece[player] == MONOMINO_PIECE_ID:
            total += MONOMINO_LAST_BONUS
    return total


def outcome(state: State) -> int:
    """+1 if player 0 wins, -1 if player 1 wins, 0 for a draw."""
    s0, s1 = score(state, 0), score(state, 1)
    if s0 > s1:
        return 1
    if s1 > s0:
        return -1
    return 0


def play_random(state: State, rng) -> State:
    """Play a full game with uniformly random legal moves (for testing)."""
    consecutive_pass = 0
    while consecutive_pass < 2:
        acts = legal_actions(state)
        if acts:
            state = apply_action(state, rng.choice(acts))
            consecutive_pass = 0
        else:
            state = apply_action(state, PASS)
            consecutive_pass += 1
    return state
