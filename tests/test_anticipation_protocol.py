import numpy as np

from scripts.anticipation import cluster_wide_times, summarize, targets


def test_targets_mark_onsets_entering_the_horizon_and_drop_repeats():
    anom = np.zeros((1, 10), bool)
    anom[0, 5:7] = True  # drained at windows 5 and 6
    fut, keep = targets(anom, np.ones_like(anom), fw=2)
    # label(t) = drained within (t, t+2]: windows 3, 4, 5
    assert np.flatnonzero(fut[0]).tolist() == [3, 4, 5]
    # rising-edge rule: rows whose previous label is already positive are dropped (4, 5, 6)
    assert np.flatnonzero(~keep[0]).tolist() == [4, 5, 6]


def test_unmonitored_windows_are_never_evaluated():
    anom = np.zeros((1, 6), bool)
    valid = np.ones_like(anom)
    valid[0, 2] = False
    _, keep = targets(anom, valid, fw=2)
    assert not keep[0, 2] and keep[0, [0, 1, 3, 4, 5]].all()


def test_cluster_wide_needs_half_of_the_eligible_nodes():
    fut = np.zeros((4, 3), bool)
    fut[:2, 1] = True  # 2 of 4 nodes at window 1
    fut[0, 2] = True  # 1 of 4 at window 2
    assert cluster_wide_times(fut, np.ones_like(fut)).tolist() == [False, True, False]


def test_summarize_splits_node_specific_from_cluster_wide_onsets():
    rng = np.random.default_rng(0)
    node = np.repeat([0, 1, 2], 10)
    y = np.tile([True, True] + [False] * 8, 3)
    s = rng.random(30) + y
    cw = np.zeros(30, bool)
    cw[:2] = True  # node 0's two onsets belong to a cluster-wide event
    res = summarize(y, s, node, cw)
    assert res["test onsets"] == 6 and res["node-specific onsets"] == 4
    assert np.isclose(res["cluster-wide share of onsets"], 1 / 3)
    assert summarize(y, s, node, cw) == res  # deterministic: own bootstrap seed
