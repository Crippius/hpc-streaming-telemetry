"""Second consumer of node-metrics: rebuild per-node feature vectors, score them with pluggable
online detectors, write every score to TimescaleDB and flagged ones to the anomaly-events topic."""

import argparse
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psycopg
from confluent_kafka import OFFSET_BEGINNING, Consumer, Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import MessageField, SerializationContext

from producer.config import load_config
from producer.melt import ANOMALY_STATES
from streaming.detectors import build

WINDOW_MS = 15 * 60 * 1000
SCHEMA_PATH = Path(__file__).parents[1] / "producer" / "schema" / "anomaly_event.avsc"
SCORE_COLUMNS = [
    "detector", "node_id", "event_time", "score", "threshold", "flagged", "nagios_state",
    "is_anomaly", "last_publish_time", "scored_at",
]  # fmt: skip
UPSERT = f"""
INSERT INTO anomaly_scores ({", ".join(SCORE_COLUMNS)})
SELECT {", ".join(SCORE_COLUMNS)} FROM stage
ON CONFLICT (detector, node_id, event_time) DO UPDATE SET
    {", ".join(f"{c} = EXCLUDED.{c}" for c in SCORE_COLUMNS[3:])}
"""


@dataclass
class Slot:
    values: np.ndarray
    seen: int = 0
    state: int | None = None
    publish_ms: int = 0


@dataclass
class Ready:
    event_time_ms: int
    vectors: dict[str, np.ndarray]
    states: dict[str, int | None]
    publish_ms: dict[str, int]


@dataclass
class Reassembler:
    """Long events -> one vector per (node, timestamp), released in timestamp order."""

    features: list[str]
    nodes: list[str]
    max_lag: int = 3
    stall_s: float = 30.0
    pending: dict[int, dict[str, Slot]] = field(default_factory=dict)
    first_seen: dict[int, float] = field(default_factory=dict)
    node_newest: dict[str, int] = field(default_factory=dict)
    node_heard: dict[str, float] = field(default_factory=dict)
    last: dict[str, np.ndarray] = field(default_factory=dict)
    released_upto: int = -1
    late: int = 0

    def __post_init__(self):
        self.index = {f: i for i, f in enumerate(self.features)}

    def add(self, ev: dict, publish_ms: int, now: float) -> None:
        t = int(ev["event_time"].timestamp() * 1000)
        i = self.index.get(f"{ev['metric']}_{ev['stat']}")
        if i is None:
            return
        node = ev["node_id"]
        self.node_newest[node] = max(self.node_newest.get(node, -1), t)
        self.node_heard[node] = now
        if t <= self.released_upto:
            self.late += 1
            return
        slots = self.pending.setdefault(t, {})
        self.first_seen.setdefault(t, now)
        slot = slots.get(node)
        if slot is None:
            slot = slots[node] = Slot(np.full(len(self.features), np.nan))
        if np.isnan(slot.values[i]):
            slot.seen += 1
        slot.values[i] = np.nan if ev["value"] is None else ev["value"]
        slot.state = ev["nagios_state"]
        slot.publish_ms = max(slot.publish_ms, publish_ms)

    def ready(self, t: int, slots: dict[str, Slot], now: float) -> bool:
        n, horizon = len(self.features), t + self.max_lag * WINDOW_MS
        for node in self.nodes:
            slot = slots.get(node)
            if (slot is None or slot.seen < n) and self.node_newest.get(node, -1) < horizon:
                heard = self.node_heard.get(node, self.first_seen[t])
                if now - heard <= self.stall_s:
                    return False
        return True

    def release(self, now: float, force: bool = False) -> list[Ready]:
        out = []
        for t in sorted(self.pending):
            if not (force or self.ready(t, self.pending[t], now)):
                break
            slots = self.pending.pop(t)
            del self.first_seen[t]
            self.released_upto = t
            out.append(self.to_ready(t, slots))
        return out

    def to_ready(self, t: int, slots: dict[str, Slot]) -> Ready:
        vectors = {}
        for node, slot in slots.items():
            x = slot.values
            prev = self.last.get(node, np.zeros_like(x))
            x = np.where(np.isnan(x), prev, x)
            vectors[node] = self.last[node] = x
        return Ready(
            t,
            vectors,
            {n: s.state for n, s in slots.items()},
            {n: s.publish_ms for n, s in slots.items()},
        )


def score_rows(name: str, detector, ready: Ready) -> list[tuple]:
    scores = detector.step(ready.vectors, ready.states)
    event_time = datetime.fromtimestamp(ready.event_time_ms / 1000, UTC)
    now = datetime.now(UTC)
    rows = []
    for node, score in scores.items():
        thr, flagged = detector.threshold.update(node, score)
        state = ready.states[node]
        rows.append((
            name, node, event_time, None if np.isnan(score) else score, thr, flagged, state,
            None if state is None else state in ANOMALY_STATES,
            datetime.fromtimestamp(ready.publish_ms[node] / 1000, UTC), now,
        ))  # fmt: skip
    return rows


def write_scores(dsn: str, rows: list[tuple]) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE stage (LIKE anomaly_scores INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        with cur.copy(f"COPY stage ({', '.join(SCORE_COLUMNS)}) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
        cur.execute(UPSERT)


def main() -> None:
    cfg = load_config()
    det_cfg = cfg["detection"]["detectors"]
    p = argparse.ArgumentParser()
    p.add_argument("--detectors", default=",".join(det_cfg), help=f"subset of {list(det_cfg)}")
    p.add_argument("--group", default="anomaly-detector")
    p.add_argument(
        "--exit-when-idle",
        type=float,
        help="stop after N idle seconds, counted from the first message (batch runs)",
    )
    p.add_argument("--pg-dsn", default="postgresql://telemetry:telemetry@localhost:5432/telemetry")
    args = p.parse_args()

    features = [f"{m}_{s}" for m in cfg["metrics"] for s in cfg["stats"]]
    detectors = {n: build(det_cfg[n], features) for n in args.detectors.split(",")}
    reasm = Reassembler(features, [str(n) for n in cfg["nodes"]])

    kcfg = cfg["kafka"]
    registry = SchemaRegistryClient({"url": kcfg["schema_registry_url"]})
    deserialize = AvroDeserializer(registry)
    in_ctx = SerializationContext(kcfg["topic"], MessageField.VALUE)
    serialize = AvroSerializer(registry, SCHEMA_PATH.read_text())
    out_ctx = SerializationContext(kcfg["anomaly_topic"], MessageField.VALUE)
    producer = Producer({"bootstrap.servers": kcfg["bootstrap_servers"], "linger.ms": 50})
    consumer = Consumer({
        "bootstrap.servers": kcfg["bootstrap_servers"],
        "group.id": args.group,
        "enable.auto.commit": False,
    })  # fmt: skip

    def from_beginning(c, partitions):
        for tp in partitions:
            tp.offset = OFFSET_BEGINNING
        c.assign(partitions)

    consumer.subscribe([kcfg["topic"]], on_assign=from_beginning)
    print(f"detectors: {list(detectors)}; {len(features)} features, {len(reasm.nodes)} nodes")

    rows, flagged_total, vectors_total = [], 0, 0
    last_flush = last_msg = last_log = time.monotonic()
    seen_any = False

    def process(ready_list):
        nonlocal flagged_total, vectors_total
        for ready in ready_list:
            vectors_total += len(ready.vectors)
            for name, det in detectors.items():
                new = score_rows(name, det, ready)
                rows.extend(new)
                for r in new:
                    if r[5]:
                        flagged_total += 1
                        event = {"detector": r[0], "node_id": r[1], "event_time": r[2],
                                 "score": r[3], "threshold": r[4], "detected_at": r[9]}  # fmt: skip
                        producer.produce(
                            kcfg["anomaly_topic"],
                            key=r[1].encode(),
                            value=serialize(event, out_ctx),
                        )

    def flush():
        nonlocal rows, last_flush
        if rows:
            write_scores(args.pg_dsn, rows)
            rows = []
        producer.poll(0)
        last_flush = time.monotonic()

    try:
        while True:
            msgs = consumer.consume(10_000, timeout=1.0)
            now = time.monotonic()
            for m in msgs:
                if m.error():
                    raise RuntimeError(m.error())
                reasm.add(deserialize(m.value(), in_ctx), m.timestamp()[1], now)
            process(reasm.release(now))
            if msgs:
                last_msg, seen_any = now, True
            elif args.exit_when_idle and seen_any and now - last_msg > args.exit_when_idle:
                break
            if len(rows) > 20_000 or now - last_flush > 2:
                flush()
            if now - last_log > 10:
                upto = datetime.fromtimestamp(max(reasm.released_upto, 0) / 1000, UTC)
                print(
                    f"scored up to {upto:%Y-%m-%d %H:%M}: {vectors_total} vectors, "
                    f"{flagged_total} flags, {reasm.late} late events ignored",
                    flush=True,
                )
                last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        process(reasm.release(time.monotonic(), force=True))
        flush()
        producer.flush()
        consumer.close()
        print(f"done: {vectors_total} vectors, {flagged_total} flags, {reasm.late} late ignored")


if __name__ == "__main__":
    main()
