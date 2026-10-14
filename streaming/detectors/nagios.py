"""Label-based references: not detectors of the telemetry, yardsticks for the ones that are."""

from producer.etl import ANOMALY_STATES
from streaming.detectors.base import Fixed, PerNodeDetector


class NagiosPersistence(PerNodeDetector):
    """Baseline: "it was anomalous 15 minutes ago". Always misses episode onsets."""

    def __init__(self, features):
        super().__init__(features, Fixed(0.5))
        self.prev = {}

    def score_learn(self, node, x, state):
        score = float(self.prev.get(node) in ANOMALY_STATES)
        self.prev[node] = state
        return score


class NagiosOracle(PerNodeDetector):
    """Scores the current label; only checks the evaluation code (must be perfect)."""

    def __init__(self, features):
        super().__init__(features, Fixed(0.5))

    def score_learn(self, node, x, state):
        return float(state in ANOMALY_STATES)
