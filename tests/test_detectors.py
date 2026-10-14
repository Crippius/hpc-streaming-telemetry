from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from streaming.detectors import build
from streaming.detectors.base import KSigma, RollingQuantile, RollingWindow
from streaming.online_detector import Reassembler

FEATURES = ["power_avg", "power_max", "temp_avg", "temp_max"]
T0 = datetime(2020, 6, 14, tzinfo=UTC)


def ev(node, minutes, feature, value, state=0):
    metric, stat = feature.rsplit("_", 1)
    return {"node_id": node, "event_time": T0 + timedelta(minutes=minutes), "metric": metric,
            "stat": stat, "value": value, "nagios_state": state}  # fmt: skip


def ms(minutes):
    return int((T0 + timedelta(minutes=minutes)).timestamp() * 1000)


def test_reassembler_releases_complete_timestamp_regardless_of_order():
    r = Reassembler(FEATURES, ["a", "b"])
    for f in reversed(FEATURES):
        r.add(ev("b", 0, f, 2.0), 1, now=0)
    for f in FEATURES[:-1]:
        r.add(ev("a", 0, f, 1.0), 1, now=0)
    assert r.release(now=0) == []
    r.add(ev("a", 0, FEATURES[-1], 1.0, state=2), 5, now=0)
    out = r.release(now=0)
    assert len(out) == 1
    assert set(out[0].vectors) == {"a", "b"} and out[0].states["a"] == 2
    assert out[0].publish_ms["a"] == 5
    assert np.allclose(out[0].vectors["b"], 2.0)


def test_reassembler_waits_for_lagging_partition():
    """Backlog catch-up: node a's partition is read days ahead of node b's. Nothing may be
    released (or discarded as late) until b catches up."""
    r = Reassembler(FEATURES, ["a", "b"])
    for m in range(0, 3 * 24 * 60, 15):
        for f in FEATURES:
            r.add(ev("a", m, f, 1.0), 0, now=0)
    assert r.release(now=0) == []
    for m in range(0, 3 * 24 * 60, 15):
        for f in FEATURES:
            r.add(ev("b", m, f, 2.0), 0, now=0)
    out = r.release(now=0)
    assert len(out) == 3 * 24 * 4 and r.late == 0
    assert all(set(o.vectors) == {"a", "b"} for o in out)


def test_reassembler_forward_fills_and_releases_stragglers_after_max_lag():
    r = Reassembler(FEATURES, ["a"], max_lag=2)
    [r.add(ev("a", 0, f, float(i)), 0, now=0) for i, f in enumerate(FEATURES)]
    r.add(ev("a", 15, "power_avg", 9.0), 0, now=0)  # t=15 incomplete: 3 features missing
    r.add(ev("a", 30, "power_avg", 9.0), 0, now=0)
    assert [o.event_time_ms for o in r.release(now=0)] == [ms(0)]
    r.add(ev("a", 45, "power_avg", 9.0), 0, now=0)  # node a is now 2 windows past t=15
    out = r.release(now=0)
    assert [o.event_time_ms for o in out] == [ms(15)]
    assert out[0].vectors["a"].tolist() == [9.0, 1.0, 2.0, 3.0]  # missing ones from t=0
    r.add(ev("a", 15, "temp_max", 1.0), 0, now=0)
    assert r.late == 1


def test_reassembler_releases_silent_node_after_stall_timeout():
    r = Reassembler(FEATURES, ["a", "b"], stall_s=30)
    for f in FEATURES:
        r.add(ev("a", 0, f, 1.0), 0, now=100)
    assert r.release(now=120) == []
    out = r.release(now=131)
    assert len(out) == 1 and set(out[0].vectors) == {"a"}


def test_reassembler_slow_but_alive_node_keeps_blocking_past_stall_timeout():
    r = Reassembler(FEATURES, ["a", "b"], stall_s=30)
    for m in range(0, 24 * 60, 15):
        for f in FEATURES:
            r.add(ev("a", m, f, 1.0), 0, now=0)
    for step, m in enumerate(range(0, 60, 15)):  # b trickles in old events, 20 s apart
        for f in FEATURES:
            r.add(ev("b", m, f, 2.0), 0, now=20 * step)
        r.release(now=20 * step)
    assert r.late == 0
    assert r.released_upto == ms(45)  # b is complete up to 00:45; 01:00 onwards keeps waiting


def test_thresholds():
    q = RollingQuantile(q=0.5, window=10, min_history=3)
    assert [q.update("a", s)[0] for s in (1, 2, 3)] == [None, None, None]
    thr, flagged = q.update("a", 10)
    assert thr == 2 and flagged
    k = KSigma(k=1, min_history=2)
    k.update("a", 1.0), k.update("a", 1.0)
    assert k.update("a", 5.0)[1]
    assert list(k.history["a"]) == [1.0, 1.0]  # flagged score kept out of the baseline


def test_rolling_window_matches_numpy():
    rng = np.random.default_rng(0)
    w = RollingWindow(maxlen=300, resync=1000)
    for i, x in enumerate(rng.normal(50, 3, 5000)):
        w.append(float(x))
        if i % 97 == 0:
            ref = np.array(list(w))
            assert np.isclose(w.mean(), ref.mean()) and np.isclose(w.std(), ref.std())
            for q in (0.5, 0.8, 0.95, 0.99):
                assert w.quantile(q) == np.quantile(ref, q)


def synthetic(n_steps=400, onset=300, seed=0):
    """3 nodes following a shared daily-ish pattern; node b goes flat and low at `onset`."""
    rng = np.random.default_rng(seed)
    base = np.sin(np.arange(n_steps) * 2 * np.pi / 96)[:, None] * [100, 120, 10, 12]
    for t in range(n_steps):
        vectors = {}
        for n in "abc":
            x = 500 + base[t] + rng.normal(0, 5, len(FEATURES))
            if n == "b" and t >= onset:
                x = np.array([200.0, 210.0, 30.0, 31.0]) + rng.normal(0, 0.01, len(FEATURES))
            vectors[n] = x
        yield t, vectors, {n: (2 if n == "b" and t >= onset else 0) for n in vectors}


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "capymoa", "algorithm": "HalfSpaceTrees", "window_size": 96},
        {"kind": "river_hst", "window_size": 96},
        {"kind": "river_hst", "window_size": 96, "peer_relative": True},
        {"kind": "relative_entropy", "window": 16, "bins": 5, "k": 2},
        {"kind": "pearson_peer", "window": 16, "k": 4},
    ],
    ids=lambda s: s["kind"] + ("_peer" if s.get("peer_relative") else ""),
)
def test_detector_scores_planted_anomaly_higher(spec):
    det = build(spec, FEATURES)
    normal, anomalous = [], []
    for t, vectors, states in synthetic():
        scores = det.step(vectors, states)
        assert set(scores) == set(vectors)
        if 200 <= t < 300 and not np.isnan(scores["b"]):
            normal.append(scores["b"])
        if 300 <= t < 320 and not np.isnan(scores["b"]):
            anomalous.append(scores["b"])
    assert normal and anomalous
    assert np.mean(anomalous) > np.mean(normal)


def test_nagios_persistence_lags_by_one_window_and_oracle_does_not():
    persist = build({"kind": "nagios_persistence"}, FEATURES)
    oracle = build({"kind": "nagios_oracle"}, FEATURES)
    x = {"a": np.zeros(len(FEATURES))}
    states = [0, 2, 2, 3, 0, None]
    assert [persist.step(x, {"a": s})["a"] for s in states] == [0, 0, 1, 1, 1, 0]
    assert [oracle.step(x, {"a": s})["a"] for s in states] == [0, 1, 1, 1, 0, 0]
