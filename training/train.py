"""Build training batches from replay samples and run one optimization step.

Loss = policy cross-entropy (target = MCTS visit distribution)
     + value_weight * value MSE (target = game result z)
L2 regularization is applied via the optimizer's weight decay.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from blokus_core.board import State
from blokus_core.encode import encode_state
from blokus_core.pieces import ACTION_SPACE


def state_from_snapshot(snap) -> State:
    occ0, occ1, used0, used1, current, num_moves = snap
    return State(occ=[occ0, occ1], used=[list(used0), list(used1)],
                 last_piece=[-1, -1], current=current,
                 finished=[False, False], num_moves=num_moves)


def make_batch(samples, device):
    """List of replay samples -> (planes, pi_target, z_target, legal_mask)."""
    n = len(samples)
    planes = np.stack([encode_state(state_from_snapshot(s[0])) for s in samples])
    pi = np.zeros((n, ACTION_SPACE), dtype=np.float32)
    legal_mask = np.zeros((n, ACTION_SPACE), dtype=bool)
    z = np.empty(n, dtype=np.float32)
    for i, (_snap, acts, probs, zz) in enumerate(samples):
        pi[i, acts] = probs
        legal_mask[i, acts] = True
        z[i] = zz
    return (torch.from_numpy(planes).to(device),
            torch.from_numpy(pi).to(device),
            torch.from_numpy(z).to(device),
            torch.from_numpy(legal_mask).to(device))


def train_step(net, optimizer, batch, value_weight: float = 1.0):
    planes, pi_target, z_target, legal_mask = batch
    net.train()
    logits, v = net(planes)
    # Train the policy only over legal moves: with ~200 legal of 17,836 actions,
    # masking stops the net from wasting capacity suppressing illegal logits.
    logits = logits.masked_fill(~legal_mask, -1e9)
    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(pi_target * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z_target)
    loss = policy_loss + value_weight * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item()),
            "policy": float(policy_loss.item()),
            "value": float(value_loss.item())}
