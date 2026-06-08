"""Policy/value network for Blokus Duo (PyTorch) + an MCTS evaluator wrapper.

Architecture: a ResNet tower over the 48 input planes, with

  * a **policy head** that produces a (NUM_VARIANTS, 14, 14) tensor and flattens
    it to ACTION_SPACE logits. The flatten order is exactly
    ``action_id = variant_id * 196 + (row * 14 + col)`` -- i.e. it matches the
    action encoding in ``pieces.py`` -- so policy targets and the legal mask line
    up with the head with no reindexing. This is far cheaper than a dense layer
    of size 6272 x 17836.
  * a **value head** producing a scalar in [-1, 1] (tanh), from the
    side-to-move's perspective.

This module requires PyTorch; the pure-Python engine (``blokus_core`` base)
does not import it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rules import BOARD_SIZE, NUM_CELLS
from .pieces import NUM_VARIANTS, ACTION_SPACE
from .encode import NUM_PLANES, encode_state

assert NUM_VARIANTS * NUM_CELLS == ACTION_SPACE  # policy-head flatten invariant


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class BlokusNet(nn.Module):
    def __init__(self, channels: int = 96, blocks: int = 8):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        # Policy head: conv -> (NUM_VARIANTS, 14, 14) -> flatten to ACTION_SPACE.
        self.p_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.p_bn = nn.BatchNorm2d(channels)
        self.p_out = nn.Conv2d(channels, NUM_VARIANTS, 1)
        # Value head.
        self.v_conv = nn.Conv2d(channels, 3, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(3)
        self.v_fc1 = nn.Linear(3 * BOARD_SIZE * BOARD_SIZE, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.tower(x)
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = self.p_out(p).flatten(1)                  # (B, ACTION_SPACE) logits
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)     # (B,)
        return p, v


def masked_policy_value(net: BlokusNet, planes: np.ndarray, legal,
                        device: str = "cpu"):
    """Single-state inference -> (probs_over_legal list, value float).

    ``legal`` is the list of legal action ids; the returned priors align to it.
    """
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    logits, v = net(x)
    logits = logits[0]
    neg_inf = torch.finfo(logits.dtype).min
    full = torch.full_like(logits, neg_inf)
    idx = torch.as_tensor(legal, dtype=torch.long, device=device)
    full[idx] = logits[idx]
    probs = torch.softmax(full, dim=0)
    priors = probs[idx].detach().cpu().numpy().tolist()
    return priors, float(v.item())


class NetEvaluator:
    """Adapts a BlokusNet to the MCTS evaluator interface."""

    def __init__(self, net: BlokusNet, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def evaluate(self, state, legal):
        return masked_policy_value(self.net, encode_state(state), legal, self.device)

    @torch.no_grad()
    def value(self, state):
        x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        _, v = self.net(x)
        return float(v.item())
