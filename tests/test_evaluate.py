import numpy as np
import pandas as pd

from scripts.evaluate_detectors import Truth, metrics, prepare

T0 = pd.Timestamp("2021-01-01", tz="UTC")


def frame(labels: dict[str, list[int]]) -> pd.DataFrame:
    rows = [
        (n, T0 + i * pd.Timedelta("15min"), lab)
        for n, seq in labels.items()
        for i, lab in enumerate(seq)
    ]
    df = pd.DataFrame(rows, columns=["node_id", "event_time", "state"])
    df["is_anomaly"] = df["state"].map({0: False, 2: True}).astype("boolean")
    df["nagios_state"], df["score"] = df["state"], 0.0
    return prepare(df.drop(columns="state"))


def test_simultaneous_onsets_in_a_rack_are_one_event():
    normal, ep = [0] * 5, [2] * 6
    df = frame({
        "200": normal + ep + normal,  # rack 10
        "201": normal + ep + normal,  # rack 10, same onset -> same event
        "221": normal + ep + normal,  # rack 11, same onset -> separate event
        "202": normal + [2, 2] + normal * 2,  # 2-window flicker: not an event
        "203": [2] * 5 + normal * 2,  # no observed onset: not an event
    })  # fmt: skip
    truth = Truth(df, T0, T0, T0 + pd.Timedelta("1D"))
    assert len(truth.episodes) == 3 and truth.n_events == 2


def test_event_delay_is_min_over_nodes_and_onset_hits():
    df = frame({"200": [0] * 5 + [2] * 6 + [0] * 5, "201": [0] * 5 + [2] * 6 + [0] * 5})
    truth = Truth(df, T0, T0, T0 + pd.Timedelta("1D"))
    flags = np.zeros(len(df), bool)
    node = df["node_id"].to_numpy()
    flags[np.flatnonzero(node == "200")[7]] = True  # 2 windows after onset
    flags[np.flatnonzero(node == "201")[6]] = True  # 1 window after onset -> event delay 1
    res = metrics(df, truth, flags, np.random.default_rng(0))
    assert res["median delay (windows)"] == 1
    assert res["events detected"].startswith("1.000") and res["onset hits"].startswith("0.000")
