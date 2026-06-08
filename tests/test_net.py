"""Policy/value network (skipped if torch is unavailable)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from blokus_core import board
from blokus_core.net import BlokusNet, NetEvaluator
from blokus_core.pieces import ACTION_SPACE
from blokus_core.encode import NUM_PLANES


def test_forward_shapes_and_value_range():
    net = BlokusNet(channels=16, blocks=2).eval()
    x = torch.zeros(4, NUM_PLANES, 14, 14)
    p, v = net(x)
    assert p.shape == (4, ACTION_SPACE)
    assert v.shape == (4,)
    assert torch.all(v <= 1.0) and torch.all(v >= -1.0)


def test_net_evaluator_priors_align_to_legal():
    ev = NetEvaluator(BlokusNet(channels=16, blocks=2), device="cpu")
    s = board.State.initial()
    legal = board.legal_actions(s)
    priors, value = ev.evaluate(s, legal)
    assert len(priors) == len(legal)
    assert abs(sum(priors) - 1.0) < 1e-4
    assert all(pr >= 0.0 for pr in priors)
    assert -1.0 <= value <= 1.0


def test_state_dict_roundtrip(tmp_path):
    net = BlokusNet(channels=16, blocks=2)
    path = tmp_path / "net.pt"
    torch.save(net.state_dict(), path)
    net2 = BlokusNet(channels=16, blocks=2)
    net2.load_state_dict(torch.load(path))
    net.eval(); net2.eval()
    x = torch.randn(1, NUM_PLANES, 14, 14)
    with torch.no_grad():
        p1, v1 = net(x)
        p2, v2 = net2(x)
    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)
