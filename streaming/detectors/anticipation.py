"""Supervised drain-onset anticipation, learned online from delayed Nagios labels."""

from collections import defaultdict, deque

import numpy as np

from producer.melt import ANOMALY_STATES
from streaming.detectors.base import Detector, RollingQuantile

LAGS = (4, 24)


class OnlineAnticipation(Detector):
    def __init__(
        self, features, horizon=4, model="river_arf", neg_sample=0.05, quantile=0.99, seed=42
    ):
        super().__init__(features, RollingQuantile(q=quantile))
        self.horizon, self.neg_sample = horizon, neg_sample
        self.rng = np.random.default_rng(seed)
        self.names = [
            f"{p}{f}" for p in ("", "d1h_", "d6h_", "vs_cluster_", "vs_rack_") for f in features
        ]
        self.model = self._make(model, seed)
        self.history = defaultdict(lambda: deque(maxlen=max(LAGS) + 1))
        self.pending = defaultdict(deque)  # node -> (step, features)
        self.states = defaultdict(deque)  # node -> (step, anomalous)
        self.prev_label = {}
        self.step_no = 0

    def _make(self, model, seed):
        if model == "river_arf":
            from river import forest

            return forest.ARFClassifier(n_models=10, seed=seed)
        if model == "river_hat":
            from river import tree

            return tree.HoeffdingAdaptiveTreeClassifier(seed=seed)
        if model == "capymoa_arf":
            from capymoa.classifier import AdaptiveRandomForestClassifier
            from capymoa.stream import Schema

            self.schema = Schema.from_custom(
                features=[*self.names, "label"], target="label", categories={"label": ["0", "1"]}
            )
            return AdaptiveRandomForestClassifier(self.schema, random_seed=seed, ensemble_size=10)
        raise ValueError(f"unknown model {model}")

    def _features(self, node, x, cluster, rack):
        h = self.history[node]
        h.append(x)
        lag = lambda k: h[-1 - k] if len(h) > k else h[0]  # noqa: E731
        return np.concatenate([x, x - lag(LAGS[0]), x - lag(LAGS[1]), x - cluster, x - rack])

    def _predict(self, f):
        if hasattr(self, "schema"):
            from capymoa.instance import Instance

            p = self.model.predict_proba(Instance.from_array(self.schema, f))
            return float(p[1]) if p is not None and len(p) > 1 else 0.0
        return self.model.predict_proba_one(dict(zip(self.names, f, strict=True))).get(True, 0.0)

    def _learn(self, f, y, p):
        if hasattr(self, "schema"):
            from capymoa.instance import LabeledInstance

            self.model.train(LabeledInstance.from_array(self.schema, f, int(y)))
        else:
            self.model.learn_one(dict(zip(self.names, f, strict=True)), y)

    def _predict_many(self, F: np.ndarray) -> np.ndarray:
        return np.array([self._predict(f) for f in F])

    def _resolve(self, node):
        """Learn every pending example of `node` whose horizon has fully elapsed."""
        pend, st = self.pending[node], self.states[node]
        while pend and pend[0][0] + self.horizon <= self.step_no:
            k, f, p = pend.popleft()
            while st and st[0][0] <= k:
                st.popleft()
            y = any(a for s, a in st if s <= k + self.horizon)
            if not self.prev_label.get(node, False) and (y or self.rng.random() < self.neg_sample):
                self._learn(f, y, p)
            self.prev_label[node] = y

    def _after_resolve(self):
        pass

    def step(self, vectors, states):
        self.step_no += 1
        nodes = list(vectors)
        values = np.stack([vectors[n] for n in nodes])
        cluster = np.median(values, axis=0)
        racks = np.array([int(n) // 20 for n in nodes])
        rack_med = {r: np.median(values[racks == r], axis=0) for r in np.unique(racks)}
        for node in nodes:
            self.states[node].append((self.step_no, states.get(node) in ANOMALY_STATES))
            self._resolve(node)
        self._after_resolve()
        F = np.stack([
            self._features(node, x, cluster, rack_med[r])
            for node, x, r in zip(nodes, values, racks, strict=True)
        ])  # fmt: skip
        p = self._predict_many(F)
        for node, f, pi in zip(nodes, F, p, strict=True):
            self.pending[node].append((self.step_no, f, pi))
        return dict(zip(nodes, p.tolist(), strict=True))


DRIFT_DETECTORS = {"optwin": "OPTWIN", "adwin": "ADWIN", "ddm": "DDM"}


class RetrainedAnticipation(OnlineAnticipation):
    """Batch gradient boosting served in the stream, refit on all resolved examples so far."""

    def __init__(self, features, horizon=4, retrain="periodic", every=2688, drift="optwin",
                 min_positives=200, min_gap=96, neg_sample=0.05, quantile=0.99, seed=42,
                 max_iter=200):  # fmt: skip
        self.retrain, self.every, self.min_gap = retrain, every, min_gap
        self.min_positives, self.max_iter, self.seed = min_positives, max_iter, seed
        self.detector = None
        if retrain == "drift":
            from capymoa.drift import detectors

            self.detector = getattr(detectors, DRIFT_DETECTORS[drift])()
        self.X_buf, self.y_buf, self.n_buf, self.events = None, None, 0, []
        self.n_pos = self.n_neg = 0
        self.drift_flag, self.last_fit = False, None
        super().__init__(features, horizon, "gbt", neg_sample, quantile, seed)

    def _make(self, model, seed):
        return None

    def _learn(self, f, y, p):
        if self.X_buf is None or self.n_buf == len(self.X_buf):  # grow by doubling
            cap = 4096 if self.X_buf is None else 2 * len(self.X_buf)
            X, Y = np.empty((cap, len(f)), np.float32), np.empty(cap, bool)
            if self.X_buf is not None:
                X[: self.n_buf], Y[: self.n_buf] = self.X_buf, self.y_buf
            self.X_buf, self.y_buf = X, Y
        self.X_buf[self.n_buf], self.y_buf[self.n_buf] = f, y
        self.n_buf += 1
        self.n_pos += y
        self.n_neg += not y
        if self.detector is not None and self.model is not None:
            if y or self.rng.random() < self.n_pos / max(self.n_neg, 1):
                self.detector.add_element(float(1 - p if y else p))
                self.drift_flag |= self.detector.detected_change()

    def _fit(self, reason):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=0.05, random_state=self.seed
        ).fit(self.X_buf[: self.n_buf], self.y_buf[: self.n_buf])
        self.last_fit, self.drift_flag = self.step_no, False
        if self.detector is not None:
            self.detector.reset()
        self.events.append({"step": self.step_no, "reason": reason, "examples": self.n_buf,
                            "positives": int(self.n_pos)})  # fmt: skip

    def _after_resolve(self):
        if self.model is None:
            if self.n_pos >= self.min_positives:
                self._fit("initial")
            return
        since = self.step_no - self.last_fit
        if self.retrain == "periodic" and since >= self.every:
            self._fit("periodic")
        elif self.retrain == "drift" and self.drift_flag and since >= self.min_gap:
            self._fit("drift")

    def _predict_many(self, F):
        if self.model is None:
            return np.zeros(len(F))
        return self.model.predict_proba(F)[:, 1]
