"""Relative entropy and Pearson correlation detectors after Rettig et al., IEEE BigData 2015,
adapted from categorical event counts to numeric per-node telemetry."""

from collections import defaultdict, deque

import numpy as np

from streaming.detectors.base import Detector, KSigma, MeanMultiple, PerNodeDetector


class RelativeEntropy(PerNodeDetector):
    """Per node and feature: KL(P || Q) between the value histograms of the last `window`
    samples (P) and the `window` samples before (Q), averaged over features. Bin edges are
    per-feature quantiles of the node's first `warmup` samples."""

    def __init__(self, features, window=16, bins=5, warmup=96, k=2.0, eps=1e-3):
        super().__init__(features, MeanMultiple(k=k))
        self.window, self.bins, self.warmup, self.eps = window, bins, warmup, eps
        self.warm = defaultdict(list)
        self.edges = {}
        self.idx = defaultdict(lambda: deque(maxlen=2 * window))

    def score_learn(self, node, x, state):
        if node not in self.edges:
            self.warm[node].append(x)
            if len(self.warm[node]) == self.warmup:
                inner = np.linspace(0, 1, self.bins + 1)[1:-1]
                self.edges[node] = np.quantile(np.array(self.warm.pop(node)), inner, axis=0).T
            return float("nan")
        buf = self.idx[node]
        buf.append((x[:, None] > self.edges[node]).sum(axis=1))
        if len(buf) < 2 * self.window:
            return float("nan")
        idx = np.array(buf)
        onehot = idx[..., None] == np.arange(self.bins)
        q = onehot[: self.window].sum(axis=0) + self.eps
        p = onehot[self.window :].sum(axis=0) + self.eps
        p, q = p / p.sum(axis=1, keepdims=True), q / q.sum(axis=1, keepdims=True)
        return float(np.mean(np.sum(p * np.log(p / q), axis=1)))


class PearsonPeer(Detector):
    """Per node and feature: Pearson r between the node's last `window` values and the cluster
    median of the same feature; score = -mean(r). A node flat while its peers move gets r = 0
    (decorrelated); features where the peers are flat too are skipped."""

    def __init__(self, features, window=16, k=4.0, min_history=96):
        super().__init__(features, KSigma(k=k, min_history=min_history))
        self.window = window
        self.buf = defaultdict(lambda: deque(maxlen=window))

    def step(self, vectors, states):
        nodes = list(vectors)
        values = np.stack([vectors[n] for n in nodes])
        median = np.median(values, axis=0)
        scores = {}
        for node, x in zip(nodes, values, strict=True):
            buf = self.buf[node]
            buf.append((x, median))
            if len(buf) < self.window:
                scores[node] = float("nan")
                continue
            a, b = (np.array(z) for z in zip(*buf, strict=True))
            a, b = a - a.mean(axis=0), b - b.mean(axis=0)
            sa, sb = np.sqrt((a * a).sum(axis=0)), np.sqrt((b * b).sum(axis=0))
            flat_a, flat_b = sa < 1e-9, sb < 1e-9
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (a * b).sum(axis=0) / (sa * sb)
            r = np.where(flat_a & ~flat_b, 0.0, r)
            r = np.where(flat_b, np.nan, r)
            scores[node] = float(-np.nanmean(r)) if np.isfinite(r).any() else float("nan")
        return scores
