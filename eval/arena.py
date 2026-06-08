"""Match arena: play games between two players and report results.

To neutralize Blokus Duo's first-player advantage, ``play_match`` plays
**color-swapped pairs**: in each pair, player A is P0 in one game and P1 in the
other. Results are reported from A's perspective, with a score rate
(wins + 0.5*draws) / games.

Players are created via zero-arg factories so each game gets a fresh player
(important for stochastic players / MCTS with their own RNGs).
"""
from __future__ import annotations

from blokus_core import board
from blokus_core.board import State, PASS


def play_game(p0, p1, first_player: int = 0, max_plies: int = 120):
    """Play one game. Returns (outcome, final_state); outcome is +1 if P0 wins."""
    s = State.initial(first_player=first_player)
    players = (p0, p1)
    consecutive_pass = 0
    plies = 0
    while consecutive_pass < 2 and plies < max_plies:
        a = players[s.current].select(s)
        consecutive_pass = consecutive_pass + 1 if a == PASS else 0
        s = board.apply_action(s, a)
        plies += 1
    return board.outcome(s), s


def play_match(make_a, make_b, num_pairs: int = 50):
    """Play ``num_pairs`` color-swapped pairs (2*num_pairs games).

    Returns a dict: a_wins, draws, b_wins, games, score_rate (A's perspective).
    """
    a_wins = draws = b_wins = 0
    for _ in range(num_pairs):
        for swap in (False, True):
            p0, p1 = (make_b(), make_a()) if swap else (make_a(), make_b())
            outcome, _ = play_game(p0, p1)
            a_res = -outcome if swap else outcome     # A's result
            if a_res > 0:
                a_wins += 1
            elif a_res < 0:
                b_wins += 1
            else:
                draws += 1
    games = a_wins + draws + b_wins
    return {
        "a_wins": a_wins, "draws": draws, "b_wins": b_wins, "games": games,
        "score_rate": (a_wins + 0.5 * draws) / games if games else 0.0,
    }
