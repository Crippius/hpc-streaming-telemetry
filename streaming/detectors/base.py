import math
from abc import ABC, abstractmethod
from bisect import bisect_left, insort
from collections import defaultdict, deque

import numpy as np


class RollingWindow:
    """Last `maxlen` values with O(1) mean/std and O(log n) quantiles.
    Running sums are recomputed exactly every `resync` appends to stop rounding drift."""

    def __init__(self, maxlen: int | None, resync: int = 10_000):
        self.values: deque = deque()
        self.sorted: list[float] = []
        self.maxlen, self.resync, self.appends = maxlen, resync, 0
        self.sum = self.sumsq = 0.0

    def append(self, x: float) -> None:
        if self.maxlen and len(self.values) == self.maxlen:
            old = self.values.popleft()
            del self.sorted[bisect_left(self.sorted, old)]
            self.sum -= old
            self.sumsq -= old * old
        self.values.append(x)
        insort(self.sorted, x)
        self.sum += x
        self.sumsq += x * x
        self.appends += 1
        if self.appends % self.resync == 0:
            self.sum = math.fsum(self.values)
            self.sumsq = math.fsum(v * v for v in self.values)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def mean(self) -> float:
        return self.sum / len(self.values)

    def std(self) -> float:
        m = self.mean()
        return math.sqrt(max(self.sumsq / len(self.values) - m * m, 0.0))

    def quantile(self, q: float) -> float:
        a = self.sorted
        pos = q * (len(a) - 1)
        lo = math.floor(pos)
        x, y, t = a[lo], a[min(lo + 1, len(a) - 1)], pos - lo
        return y - (y - x) * (1 - t) if t >= 0.5 else x + (y - x) * t


class Threshold(ABC):
    """Per-node threshold computed from the node's own past scores (the current one excluded)."""

    maxlen: int | None = None
    learn_when_flagged = True

    def __init__(self, min_history: int = 96):
        self.min_history = min_history
        self.history: dict[str, RollingWindow] = defaultdict(lambda: RollingWindow(self.maxlen))

    @abstractmethod
    def value(self, past: RollingWindow) -> float: ...

    def update(self, node: str, score: float) -> tuple[float | None, bool]:
        if math.isnan(score):
            return None, False
        past = self.history[node]
        thr = self.value(past) if len(past) >= self.min_history else None
        flagged = thr is not None and score > thr
        if self.learn_when_flagged or not flagged:
            past.append(score)
        return thr, flagged


class Fixed(Threshold):
    def __init__(self, value: float):
        super().__init__(min_history=0)
        self._value = value

    def value(self, past):
        return self._value


class RollingQuantile(Threshold):
    def __init__(self, q: float = 0.95, window: int = 500, min_history: int = 96):
        self.q, self.maxlen = q, window
        super().__init__(min_history)

    def value(self, past):
        return past.quantile(self.q)


class MeanMultiple(Threshold):
    """flag when score > k * baseline mean"""

    maxlen, learn_when_flagged = 2000, False

    def __init__(self, k: float = 2.0, min_history: int = 96):
        self.k = k
        super().__init__(min_history)

    def value(self, past):
        return self.k * past.mean()


class KSigma(Threshold):
    """flag when score > baseline mean + k * std"""

    maxlen, learn_when_flagged = 2000, False

    def __init__(self, k: float = 4.0, min_history: int = 96):
        self.k = k
        super().__init__(min_history)

    def value(self, past):
        return past.mean() + self.k * past.std()


class Detector(ABC):
    """Scores one timestamp of the cluster at a time, prequentially (score, then learn).

    `vectors` maps node_id -> feature vector (order = `features`), `states` maps node_id -> raw
    Nagios state. Per-node models loop over nodes; cross-node detectors can use the peers.
    Scores: higher = more anomalous; NaN = no opinion yet (warm-up).
    """

    def __init__(self, features: list[str], threshold: Threshold):
        self.features = features
        self.threshold = threshold

    @abstractmethod
    def step(
        self, vectors: dict[str, np.ndarray], states: dict[str, int | None]
    ) -> dict[str, float]: ...


class PerNodeDetector(Detector):
    def step(self, vectors, states):
        return {n: self.score_learn(n, x, states.get(n)) for n, x in vectors.items()}

    @abstractmethod
    def score_learn(self, node: str, x: np.ndarray, state: int | None) -> float: ...


class RunningMinMax:
    """Online min-max scaling with past statistics."""

    def __init__(self):
        self.lo = self.hi = None

    def transform_learn(self, x: np.ndarray) -> np.ndarray:
        if self.lo is None:
            self.lo, self.hi = x.copy(), x.copy()
        span = self.hi - self.lo
        out = np.divide(x - self.lo, span, out=np.zeros_like(x), where=span > 0)
        self.lo, self.hi = np.minimum(self.lo, x), np.maximum(self.hi, x)
        return out


class PeerRelative(Detector):
    """Feeds the wrapped detector each node's deviation from the cluster median at the same
    timestamp, so a node idle while its peers are busy stands out even though idle is normal."""

    def __init__(self, inner: Detector):
        super().__init__(inner.features, inner.threshold)
        self.inner = inner

    def step(self, vectors, states):
        median = np.median(np.stack(list(vectors.values())), axis=0)
        return self.inner.step({n: x - median for n, x in vectors.items()}, states)
