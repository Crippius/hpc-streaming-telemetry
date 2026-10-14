import numpy as np
import pandas as pd
import pytest

from producer.etl import LONG_COLUMNS, label_to_anomaly, to_events

METRICS = ["total_power", "ambient"]
STATS = ["avg", "std", "min", "max"]


@pytest.fixture
def wide() -> pd.DataFrame:
    ts = pd.date_range("2020-06-14", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "value": pd.array([0, 2, 3, None], dtype="Int32")})
    for m in [*METRICS, "pcie"]:
        for s in STATS:
            df[f"{m}_{s}"] = np.arange(4, dtype="float32")
    df.loc[0, "ambient_avg"] = np.nan
    return df


def test_row_count_and_schema(wide):
    long = to_events(wide, "215", METRICS, STATS)
    assert len(long) == len(wide) * len(METRICS) * len(STATS)
    assert list(long.columns) == LONG_COLUMNS
    assert set(long["metric"]) == set(METRICS)  # unselected metric (pcie) dropped
    assert set(long["stat"]) == set(STATS)
    assert (long["node_id"] == "215").all()
    assert str(long["event_time"].dtype) == "datetime64[ms, UTC]"


def test_values_follow_their_column(wide):
    long = to_events(wide, "215", METRICS, STATS)
    row = long[(long["metric"] == "total_power") & (long["stat"] == "max")]
    assert row["value"].tolist() == [0.0, 1.0, 2.0, 3.0]
    nan = long[(long["metric"] == "ambient") & (long["stat"] == "avg")]["value"]
    assert nan.isna().tolist() == [True, False, False, False]


def test_label_mapping(wide):
    long = to_events(wide, "215", METRICS, STATS)
    per_ts = long.groupby("event_time").first()
    assert per_ts["nagios_state"].tolist()[:3] == [0, 2, 3]
    assert pd.isna(per_ts["nagios_state"].iloc[3])
    assert per_ts["is_anomaly"].tolist()[:3] == [False, True, True]
    assert pd.isna(per_ts["is_anomaly"].iloc[3])


def test_label_to_anomaly_nullable():
    out = label_to_anomaly(pd.Series(pd.array([0, 2, 3, None], dtype="Int32")))
    assert str(out.dtype) == "boolean"
    assert out.tolist()[:3] == [False, True, True] and pd.isna(out.iloc[3])


def test_unknown_metric_is_clear_error(wide):
    with pytest.raises(ValueError, match=r"unknown metric\(s\) \['p0_core99_temp'\]"):
        to_events(wide, "215", [*METRICS, "p0_core99_temp"], STATS)
