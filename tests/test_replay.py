"""Replay buffer: ring behavior and recap (drop stale data on resume)."""
from __future__ import annotations

from training.replay import ReplayBuffer


def test_ring_overwrites_oldest():
    b = ReplayBuffer(3)
    for x in (1, 2, 3, 4):
        b.add(x)
    assert len(b) == 3
    assert set(b.data) == {2, 3, 4}        # 1 evicted


def test_recap_keeps_most_recent():
    b = ReplayBuffer(100)
    for i in range(80):
        b.add(i)
    b.recap(50)
    assert b.capacity == 50
    assert len(b.data) == 50
    assert b.data[0] == 30 and b.data[-1] == 79   # kept the newest 50
    b.add(999)                                     # overwrites the oldest slot
    assert len(b.data) == 50 and 999 in b.data
