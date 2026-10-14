import json

import fastavro
import numpy as np
import pandas as pd

from producer.replay_producer import SCHEMA_PATH, WINDOW, schedule, to_records


def events(n_windows=6, nodes=("200", "201")) -> pd.DataFrame:
    ts = pd.date_range("2020-06-14", periods=n_windows, freq="15min", tz="UTC").astype(
        "datetime64[ms, UTC]"
    )
    df = pd.DataFrame(
        [(n, t, "total_power", "avg", float(i)) for i, t in enumerate(ts) for n in nodes],
        columns=["node_id", "event_time", "metric", "stat", "value"],
    )
    df["nagios_state"] = pd.array([0, 2] * (len(df) // 2), dtype="Int32")
    df["is_anomaly"] = pd.array([False, True] * (len(df) // 2), dtype="boolean")
    return df


def test_schedule_without_late_keeps_event_order():
    df = events()
    out = schedule(df, late_fraction=0.0, late_delay=2)
    assert not out["late"].any()
    assert (out["emit_time"] == out["event_time"]).all()
    assert out["event_time"].is_monotonic_increasing


def test_schedule_delays_late_events_by_whole_windows():
    out = schedule(events(n_windows=40), late_fraction=0.2, late_delay=3, seed=1)
    late = out[out["late"]]
    assert 0 < len(late) < len(out)
    assert ((late["emit_time"] - late["event_time"]) == 3 * WINDOW).all()
    assert out["emit_time"].is_monotonic_increasing
    # a late event is emitted after on-time events with a later event_time
    first_late = out.index[out["late"]][0]
    assert (out.loc[: first_late - 1, "event_time"] > out.loc[first_late, "event_time"]).any()


def test_schedule_is_deterministic():
    a = schedule(events(n_windows=40), 0.2, 2, seed=7)
    b = schedule(events(n_windows=40), 0.2, 2, seed=7)
    assert np.array_equal(a["late"], b["late"])


def test_records_match_avro_schema_with_nulls():
    df = events(n_windows=2)
    df.loc[0, "value"] = np.nan
    df.loc[1, ["nagios_state", "is_anomaly"]] = pd.NA
    recs = to_records(df)
    schema = fastavro.parse_schema(json.loads(SCHEMA_PATH.read_text()))
    for r in recs:
        assert fastavro.validate(r, schema)
    assert recs[0]["value"] is None
    assert recs[1]["nagios_state"] is None and recs[1]["is_anomaly"] is None
    assert recs[2]["is_anomaly"] is False and isinstance(recs[2]["nagios_state"], int)
