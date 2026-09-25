"""Replay the melted M100 events into Kafka in event-time order, compressed in wall-clock time."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import MessageField, SerializationContext

from producer.config import ROOT, load_config

SCHEMA_PATH = Path(__file__).parent / "schema" / "metric_event.avsc"
WINDOW = pd.Timedelta("15min")
FIELDS = ["node_id", "event_time", "metric", "stat", "value", "nagios_state", "is_anomaly"]


def load_events(long_dir: Path, nodes: list[str], max_windows: int | None) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(long_dir / f"{n}.parquet") for n in nodes], ignore_index=True)
    if max_windows:
        first = df["event_time"].min()
        df = df[df["event_time"] < first + max_windows * WINDOW]
    return df.sort_values("event_time", kind="stable", ignore_index=True)


def schedule(
    df: pd.DataFrame, late_fraction: float, late_delay: int, seed: int = 0
) -> pd.DataFrame:
    """Add `emit_time`: event_time, or event_time + late_delay windows for injected late events."""
    rng = np.random.default_rng(seed)
    late = rng.random(len(df)) < late_fraction
    df = df.assign(late=late, emit_time=df["event_time"] + late * late_delay * WINDOW)
    return df.sort_values("emit_time", kind="stable", ignore_index=True)


def to_records(df: pd.DataFrame) -> list[dict]:
    cols = {
        "node_id": df["node_id"].tolist(),
        "event_time": df["event_time"].dt.to_pydatetime().tolist(),
        "metric": df["metric"].tolist(),
        "stat": df["stat"].tolist(),
        "value": df["value"].astype(object).where(df["value"].notna(), None).tolist(),
        "nagios_state": [None if pd.isna(v) else int(v) for v in df["nagios_state"]],
        "is_anomaly": [None if pd.isna(v) else bool(v) for v in df["is_anomaly"]],
    }
    return [dict(zip(FIELDS, row, strict=True)) for row in zip(*cols.values(), strict=True)]


def log_late(df: pd.DataFrame, path: Path) -> None:
    late = df[df["late"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "event_time", "metric", "stat", "emitted_after"])
        for r in late.itertuples():
            w.writerow(
                [r.node_id, r.event_time.isoformat(), r.metric, r.stat, r.emit_time.isoformat()]
            )
    print(f"injected {len(late)} late events -> {path.relative_to(ROOT)}")


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--windows-per-sec",
        type=float,
        default=cfg["replay"]["windows_per_sec"],
        help="15-min windows replayed per wall-clock second; 0 = as fast as possible",
    )
    p.add_argument("--nodes", help="comma-separated subset of config nodes")
    p.add_argument("--max-windows", type=int, help="replay only the first N windows")
    p.add_argument("--late-fraction", type=float, default=0.0)
    p.add_argument("--late-delay", type=int, default=2, help="delay of late events, in windows")
    args = p.parse_args()

    nodes = args.nodes.split(",") if args.nodes else [str(n) for n in cfg["nodes"]]
    df = schedule(
        load_events(cfg["long_dir"], nodes, args.max_windows), args.late_fraction, args.late_delay
    )
    if args.late_fraction:
        log_late(df, cfg["long_dir"].parent / "late_events.csv")
    ticks = df["emit_time"].to_numpy()
    tick_starts = np.flatnonzero(np.r_[True, ticks[1:] != ticks[:-1]])
    records = to_records(df)
    print(f"{len(records)} events, {len(nodes)} nodes, {len(tick_starts)} windows")

    kcfg = cfg["kafka"]
    topic = kcfg["topic"]
    serialize = AvroSerializer(
        SchemaRegistryClient({"url": kcfg["schema_registry_url"]}), SCHEMA_PATH.read_text()
    )
    ctx = SerializationContext(topic, MessageField.VALUE)
    producer = Producer(
        {
            "bootstrap.servers": kcfg["bootstrap_servers"],
            "linger.ms": 20,
            "batch.size": 1_000_000,
            "compression.type": "lz4",
            "queue.buffering.max.messages": 1_000_000,
        }
    )

    bounds = [*tick_starts, len(records)]
    t0 = last_log = time.perf_counter()
    sent_at_log = 0
    for i in range(len(tick_starts)):
        if args.windows_per_sec:
            delay = t0 + i / args.windows_per_sec - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        for rec in records[bounds[i] : bounds[i + 1]]:
            payload = serialize(rec, ctx)
            while True:
                try:
                    producer.produce(topic, key=rec["node_id"].encode(), value=payload)
                    break
                except BufferError:
                    producer.poll(0.05)
        producer.poll(0)
        now = time.perf_counter()
        if now - last_log >= 5:
            rate = (bounds[i + 1] - sent_at_log) / (now - last_log)
            print(f"{ticks[bounds[i]]}  window {i + 1}/{len(tick_starts)}  {rate:,.0f} msg/s")
            last_log, sent_at_log = now, bounds[i + 1]

    producer.flush()
    elapsed = time.perf_counter() - t0
    print(f"done: {len(records)} msgs in {elapsed:.1f}s ({len(records) / elapsed:,.0f} msg/s avg)")


if __name__ == "__main__":
    main()
