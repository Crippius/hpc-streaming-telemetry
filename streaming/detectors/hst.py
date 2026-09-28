"""Half-Space Trees from two streaming-ML libraries, one model per node."""

from collections import defaultdict

import numpy as np

from streaming.detectors.base import PerNodeDetector, RollingQuantile, RunningMinMax


class CapyMOA(PerNodeDetector):
    def __init__(self, features, algorithm="HalfSpaceTrees", seed=42, quantile=0.95, **params):
        from capymoa import anomaly
        from capymoa.stream import Schema

        super().__init__(features, RollingQuantile(q=quantile))
        self.schema = Schema.from_custom(
            features=[*features, "label"], target="label", categories={"label": ["0", "1"]}
        )
        self.make = lambda: getattr(anomaly, algorithm)(
            schema=self.schema, random_seed=seed, **params
        )
        self.models = {}
        self.scalers = defaultdict(RunningMinMax)

    def score_learn(self, node, x, state):
        from capymoa.instance import LabeledInstance

        if node not in self.models:
            self.models[node] = self.make()
        model = self.models[node]
        # MOA's one-class learners ignore unlabelled instances: train everything as class 0
        inst = LabeledInstance.from_array(self.schema, self.scalers[node].transform_learn(x), 0)
        score = model.score_instance(inst)
        model.train(inst)
        return float(score)


class RiverHST(PerNodeDetector):
    def __init__(self, features, seed=42, quantile=0.95, **params):
        from river import anomaly, compose, preprocessing

        super().__init__(features, RollingQuantile(q=quantile))
        self.make = lambda: compose.Pipeline(
            preprocessing.MinMaxScaler(), anomaly.HalfSpaceTrees(seed=seed, **params)
        )
        self.models = {}

    def score_learn(self, node, x, state):
        if node not in self.models:
            self.models[node] = self.make()
        model = self.models[node]
        xd = dict(zip(self.features, np.asarray(x, dtype=float).tolist(), strict=True))
        score = model.score_one(xd)
        model.learn_one(xd)
        return float(score)
