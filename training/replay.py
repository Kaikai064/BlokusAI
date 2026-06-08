"""FIFO replay buffer of self-play samples.

Each sample is ``(snapshot, action_ids, pi, z)``:
  * snapshot  : compact State fields, planes are reconstructed at sample time
                (keeps the buffer ~10x smaller than storing dense planes)
  * action_ids: int32 array of the legal actions at that position
  * pi        : float32 MCTS visit-count distribution over those actions
  * z         : game result in [-1, 1] from that position's side-to-move
"""
from __future__ import annotations

import os
import pickle


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []
        self.pos = 0

    def add(self, sample) -> None:
        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            self.data[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def extend(self, samples) -> None:
        for s in samples:
            self.add(s)

    def sample(self, n: int, rng):
        m = len(self.data)
        return [self.data[rng.randrange(m)] for _ in range(n)]

    def __len__(self) -> int:
        return len(self.data)

    def recap(self, new_capacity: int) -> None:
        """Resize capacity, keeping the most recent samples. Used on resume to
        drop stale early data when ``buffer_size`` is lowered."""
        self.capacity = new_capacity
        if len(self.data) > new_capacity:
            self.data = self.data[-new_capacity:]
        self.pos = 0

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"capacity": self.capacity, "data": self.data, "pos": self.pos},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.capacity = d["capacity"]
        self.data = d["data"]
        self.pos = d["pos"]
