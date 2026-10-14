import numpy as np

from streaming.detectors import build

FEATURES = ["power_avg", "temp_avg"]


class Recorder:
    def __init__(self):
        self.learned = []

    def predict_proba_one(self, x):
        return {True: 0.0}

    def learn_one(self, x, y):
        self.learned.append((x["power_avg"], y))


def run(states_seq, horizon=2):
    det = build({"kind": "anticipation", "horizon": horizon, "neg_sample": 1.0}, FEATURES)
    det.model = Recorder()
    for t, s in enumerate(states_seq):
        det.step({"200": np.array([float(t), 0.0])}, {"200": s})
    return det.model.learned


def test_labels_come_from_the_next_horizon_steps():
    # step value t is the feature, drain (state 2) at t = 4
    learned = dict(run([0, 0, 0, 0, 2, 0, 0, 0, 0, 0]))
    assert learned[1.0] is False  # (1, 3] has no drain
    assert learned[2.0] is True  # (2, 4] contains t = 4


def test_rising_edge_rule_skips_examples_after_a_positive():
    learned = [t for t, _ in run([0, 0, 0, 0, 2, 0, 0, 0, 0, 0])]
    assert 2.0 in learned and 3.0 not in learned  # 3 follows positive example 2
    assert 4.0 not in learned  # 4 follows positive example 3 (label of 3 is also positive)


def test_every_example_is_learned_once_horizon_has_elapsed():
    learned = run([0] * 10, horizon=3)
    assert [t for t, _ in learned] == [float(t) for t in range(7)]  # steps 0..6 resolved by 9
    assert not any(y for _, y in learned)


def gbt_run(steps=120, **params):
    spec = {"kind": "anticipation_gbt", "horizon": 2, "min_positives": 2, "neg_sample": 1.0,
            "max_iter": 5, **params}  # fmt: skip
    det = build(spec, FEATURES)
    rng = np.random.default_rng(0)
    for t in range(steps):
        drain = t % 10 in (8, 9)  # regular drains -> plenty of positives
        vectors = {
            n: rng.normal(0, 1, len(FEATURES)) + (3.0 if t % 10 == 7 else 0) for n in ("200", "201")
        }
        det.step(vectors, {n: 2 if drain else 0 for n in vectors})
    return det


def test_retrain_none_fits_exactly_once():
    det = gbt_run(retrain="none")
    assert [e["reason"] for e in det.events] == ["initial"]


def test_retrain_periodic_refits_on_schedule():
    det = gbt_run(retrain="periodic", every=30)
    steps = [e["step"] for e in det.events]
    assert det.events[0]["reason"] == "initial" and len(steps) >= 3
    assert all(b - a >= 30 for a, b in zip(steps, steps[1:], strict=False))


class AlwaysDrift:
    def add_element(self, x):
        pass

    def detected_change(self):
        return True

    def reset(self):
        pass


def test_retrain_drift_refits_on_detection_with_min_gap():
    det = build({"kind": "anticipation_gbt", "horizon": 2, "min_positives": 2, "neg_sample": 1.0,
                 "max_iter": 5, "retrain": "drift", "min_gap": 25}, FEATURES)  # fmt: skip
    det.detector = AlwaysDrift()
    rng = np.random.default_rng(0)
    for t in range(120):
        vectors = {n: rng.normal(0, 1, len(FEATURES)) for n in ("200", "201")}
        det.step(vectors, {n: 2 if t % 10 in (8, 9) else 0 for n in vectors})
    reasons = [e["reason"] for e in det.events]
    assert reasons[0] == "initial" and "drift" in reasons
    steps = [e["step"] for e in det.events]
    assert all(b - a >= 25 for a, b in zip(steps, steps[1:], strict=False))
