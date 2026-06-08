"""AlphaZero-style MCTS (PUCT) with a pluggable position evaluator.

The same search is used for self-play (with Dirichlet root noise and a sampling
temperature) and for match/GUI play (noiseless, low temperature). The evaluator
is any object with:

    evaluate(state, legal) -> (priors_aligned_to_legal, value_in_[-1,1])
    value(state)          -> value_in_[-1,1]      # used at pass-only nodes

Values are always from the perspective of the side to move at that node;
back-ups negate per ply (negamax). This module is torch-free; plug in a
``NetEvaluator`` (see net.py) to use the neural network.
"""
from __future__ import annotations

import math
import random

import numpy as np

from . import board


class UniformEvaluator:
    """Uniform priors, neutral value. A weak evaluator useful for tests."""

    def evaluate(self, state, legal):
        n = len(legal)
        return [1.0 / n] * n, 0.0

    def value(self, state):
        return 0.0


class RolloutEvaluator:
    """Pure-MCTS evaluator: uniform priors, value from random rollouts.

    Slow in pure Python (each rollout plays a full random game) but is the
    classic 'search without learned knowledge' baseline (ladder rung B4).
    """

    def __init__(self, rng=None, n_rollouts: int = 1):
        self.rng = rng or random.Random()
        self.n_rollouts = n_rollouts

    def _rollout_value(self, state):
        end = board.play_random(state, self.rng)
        o = board.outcome(end)
        return float(o if state.current == 0 else -o)

    def evaluate(self, state, legal):
        n = len(legal)
        total = sum(self._rollout_value(state) for _ in range(self.n_rollouts))
        return [1.0 / n] * n, total / self.n_rollouts

    def value(self, state):
        return self._rollout_value(state)


class _Node:
    __slots__ = ("state", "legal", "priors", "N", "W", "Q",
                 "children", "is_terminal", "expanded")

    def __init__(self, state):
        self.state = state
        self.legal = None
        self.priors = None
        self.N = self.W = self.Q = None
        self.children = None
        self.is_terminal = False
        self.expanded = False


def _terminal_value(state, player):
    o = board.outcome(state)
    return float(o if player == 0 else -o)


class MCTS:
    def __init__(self, evaluator, n_sims: int = 128, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 rng=None):
        self.ev = evaluator
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha
        self.eps = dirichlet_eps
        self.rng = rng or random.Random()
        self.np_rng = np.random.default_rng(self.rng.randrange(2 ** 32))

    def _expand(self, node):
        s = node.state
        if board.is_terminal(s):
            node.is_terminal = True
            node.expanded = True
            return _terminal_value(s, s.current)
        legal = board.legal_actions(s)
        if not legal:
            node.legal = [board.PASS]
            node.priors = np.array([1.0], dtype=np.float64)
            value = self.ev.value(s)
        else:
            priors, value = self.ev.evaluate(s, legal)
            node.legal = legal
            node.priors = np.asarray(priors, dtype=np.float64)
            total = node.priors.sum()
            node.priors = node.priors / total if total > 0 else np.full(
                len(legal), 1.0 / len(legal))
        k = len(node.legal)
        node.N = np.zeros(k)
        node.W = np.zeros(k)
        node.Q = np.zeros(k)
        node.children = [None] * k
        node.expanded = True
        return value

    def _simulate(self, node):
        if node.is_terminal:
            return _terminal_value(node.state, node.state.current)
        if not node.expanded:
            return self._expand(node)
        sqrt_total = math.sqrt(node.N.sum() + 1.0)
        u = node.Q + self.c_puct * node.priors * sqrt_total / (1.0 + node.N)
        i = int(np.argmax(u))
        if node.children[i] is None:
            node.children[i] = _Node(board.apply_action(node.state, node.legal[i]))
        v = -self._simulate(node.children[i])
        node.N[i] += 1
        node.W[i] += v
        node.Q[i] = node.W[i] / node.N[i]
        return v

    def run(self, state, add_noise: bool = False) -> _Node:
        root = _Node(state.copy())
        self._expand(root)
        if add_noise and root.legal and len(root.legal) > 1:
            noise = self.np_rng.dirichlet([self.alpha] * len(root.legal))
            root.priors = (1.0 - self.eps) * root.priors + self.eps * noise
        for _ in range(self.n_sims):
            self._simulate(root)
        return root

    def policy(self, state, temperature: float = 1.0, add_noise: bool = False):
        """Return (legal_actions, probabilities) from MCTS visit counts."""
        root = self.run(state, add_noise=add_noise)
        counts = root.N.astype(np.float64)
        if counts.sum() == 0:
            probs = root.priors.copy()
        elif temperature <= 1e-3:
            probs = np.zeros_like(counts)
            probs[int(np.argmax(counts))] = 1.0
        else:
            scaled = counts ** (1.0 / temperature)
            probs = scaled / scaled.sum()
        return list(root.legal), probs


class MCTSPlayer:
    """Adapts MCTS to the arena's ``select(state) -> action`` interface."""

    def __init__(self, evaluator, n_sims: int = 128, c_puct: float = 1.5,
                 temperature: float = 0.0, add_noise: bool = False, rng=None):
        self.mcts = MCTS(evaluator, n_sims=n_sims, c_puct=c_puct, rng=rng)
        self.temperature = temperature
        self.add_noise = add_noise

    def select(self, state):
        if not board.legal_actions(state):
            return board.PASS
        legal, probs = self.mcts.policy(
            state, temperature=self.temperature, add_noise=self.add_noise)
        if self.temperature <= 1e-3:
            return legal[int(np.argmax(probs))]
        return legal[int(self.mcts.np_rng.choice(len(legal), p=probs))]
