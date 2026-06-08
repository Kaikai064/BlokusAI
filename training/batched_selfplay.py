"""Batched self-play: run many games in lockstep so their MCTS leaf evaluations
fuse into single GPU batches -- the main self-play throughput lever.

Each round of search, every active game contributes exactly ONE leaf, so all
leaves are evaluated in one network forward pass and no virtual loss is needed.
The search takes a pluggable *batch evaluator* (``evaluate_batch(states, legals)
-> (priors_list, values)``); with ``UniformBatchEvaluator`` and one game it
reproduces the recursive ``blokus_core.mcts`` search exactly (see tests), which
is how the batched implementation is validated.
"""
from __future__ import annotations

import math

import numpy as np

from blokus_core import board
from blokus_core.board import State
from blokus_core.encode import encode_state
from .selfplay import snapshot


# --- batch evaluators --------------------------------------------------------
class UniformBatchEvaluator:
    """Uniform priors, neutral value. Matches mcts.UniformEvaluator, batched."""

    def evaluate_batch(self, states, legals):
        priors = [np.ones(len(L), dtype=np.float64) / len(L) for L in legals]
        return priors, [0.0] * len(states)


class NetBatchEvaluator:
    """Batched neural-net evaluator. One forward pass for a list of states."""

    def __init__(self, net, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device

    def evaluate_batch(self, states, legals):
        import torch
        planes = np.stack([encode_state(s) for s in states])
        x = torch.from_numpy(planes).to(self.device)
        with torch.no_grad():
            logits, values = self.net(x)
        logits = logits.detach().cpu().numpy()
        values = values.detach().cpu().numpy()
        priors = []
        for row, legal in enumerate(legals):
            if len(legal) == 1 and legal[0] == board.PASS:
                priors.append(np.array([1.0]))
            else:
                ll = logits[row][legal]
                ll = ll - ll.max()
                e = np.exp(ll)
                priors.append(e / e.sum())
        return priors, [float(v) for v in values]


# --- tree --------------------------------------------------------------------
class _Node:
    __slots__ = ("state", "legal", "priors", "N", "W", "Q", "children",
                 "is_terminal", "expanded", "pending_legal")

    def __init__(self, state):
        self.state = state
        self.legal = self.priors = self.N = self.W = self.Q = None
        self.children = None
        self.is_terminal = False
        self.expanded = False
        self.pending_legal = None


def _terminal_value(state, player):
    o = board.outcome(state)
    return float(o if player == 0 else -o)


def _init_node(node, legal, priors):
    total = priors.sum()
    node.legal = legal
    node.priors = priors / total if total > 0 else np.full(len(legal), 1.0 / len(legal))
    k = len(legal)
    node.N = np.zeros(k)
    node.W = np.zeros(k)
    node.Q = np.zeros(k)
    node.children = [None] * k
    node.expanded = True


def _backup(path, leaf_value):
    v = leaf_value
    for parent, i in reversed(path):
        v = -v
        parent.N[i] += 1
        parent.W[i] += v
        parent.Q[i] = parent.W[i] / parent.N[i]


def _select(node, c_puct):
    """Descend to a leaf. Returns (leaf, path) needing evaluation, or None if a
    terminal leaf was reached and already backed up."""
    path = []
    while node.expanded and not node.is_terminal:
        u = node.Q + c_puct * node.priors * math.sqrt(node.N.sum() + 1.0) / (1.0 + node.N)
        i = int(np.argmax(u))
        path.append((node, i))
        if node.children[i] is None:
            node.children[i] = _Node(board.apply_action(node.state, node.legal[i]))
        node = node.children[i]
    if node.is_terminal:
        _backup(path, _terminal_value(node.state, node.state.current))
        return None
    legal = board.legal_actions(node.state)
    if not legal:
        if not board.has_legal(node.state, 1 - node.state.current):
            node.is_terminal = True
            node.expanded = True
            _backup(path, _terminal_value(node.state, node.state.current))
            return None
        node.pending_legal = [board.PASS]
    else:
        node.pending_legal = legal
    return node, path


def search_batch(states, batch_ev, n_sims, c_puct=1.5, np_rng=None,
                 add_noise=False, alpha=0.3, eps=0.25):
    """Run MCTS for a list of root states in lockstep. Returns the root nodes."""
    roots = [_Node(s) for s in states]
    legals = [board.legal_actions(r.state) for r in roots]
    priors_list, _ = batch_ev.evaluate_batch([r.state for r in roots], legals)
    for r, legal, pr in zip(roots, legals, priors_list):
        pr = np.asarray(pr, dtype=np.float64)
        if add_noise and np_rng is not None and len(legal) > 1:
            noise = np_rng.dirichlet([alpha] * len(legal))
            pr = (1.0 - eps) * pr + eps * noise
        _init_node(r, legal, pr)

    for _ in range(n_sims):
        nodes, paths, states_b, legals_b = [], [], [], []
        for r in roots:
            leaf = _select(r, c_puct)
            if leaf is not None:
                node, path = leaf
                nodes.append(node)
                paths.append(path)
                states_b.append(node.state)
                legals_b.append(node.pending_legal)
        if nodes:
            priors_list, values = batch_ev.evaluate_batch(states_b, legals_b)
            for node, path, pr, v in zip(nodes, paths, priors_list, values):
                _init_node(node, node.pending_legal, np.asarray(pr, dtype=np.float64))
                _backup(path, float(v))
    return roots


def generate_games(batch_ev, cfg, rng, num_games):
    """Play ``num_games`` self-play games concurrently. Returns training samples."""
    np_rng = np.random.default_rng(rng.randrange(2 ** 32))
    games = [{"state": State.initial(), "records": [], "done": False, "ply": 0}
             for _ in range(num_games)]

    rounds = 0
    while rounds <= cfg.max_plies:
        to_search = []
        for g in games:
            if g["done"]:
                continue
            while not g["done"]:                      # resolve passes / terminal
                if board.is_terminal(g["state"]):
                    g["done"] = True
                elif board.legal_actions(g["state"]):
                    break
                else:
                    g["state"] = board.apply_action(g["state"], board.PASS)
            if not g["done"]:
                to_search.append(g)
        if not to_search:
            break

        roots = search_batch([g["state"].copy() for g in to_search], batch_ev,
                             cfg.n_sims, cfg.c_puct, np_rng=np_rng, add_noise=True,
                             alpha=cfg.dirichlet_alpha, eps=cfg.dirichlet_eps)
        for g, root in zip(to_search, roots):
            counts = root.N.astype(np.float64)
            if counts.sum() == 0:
                pi = np.ones(len(root.legal)) / len(root.legal)
                a_idx = 0
            else:
                pi = counts / counts.sum()
                a_idx = (int(np_rng.choice(len(counts), p=pi))
                         if g["ply"] < cfg.temp_moves else int(np.argmax(counts)))
            g["records"].append((snapshot(g["state"]),
                                 np.asarray(root.legal, dtype=np.int32),
                                 pi.astype(np.float32), g["state"].current))
            g["state"] = board.apply_action(g["state"], root.legal[a_idx])
            g["ply"] += 1
        rounds += 1

    samples = []
    for g in games:
        z = board.outcome(g["state"])
        for snap, acts, pi, player in g["records"]:
            samples.append((snap, acts, pi, float(z if player == 0 else -z)))
    return samples
