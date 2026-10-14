"""Batch mode: run the stream detectors over full node histories from rack tarballs."""

import argparse
import io
import json
import tarfile
import time
from multiprocessing import Process

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from producer.config import load_config
from producer.etl import ANOMALY_STATES
from streaming.detectors import build

STEP = pd.Timedelta("15min")


def cache_dir(cfg):
    return cfg["long_dir"].parent / "offline"


def load_racks(cfg, racks: list[str], features: list[str]) -> None:
    """Wide node files -> X[node, t, feature] (float32, forward-filled per node), states[node, t]
    (int8, -1 = NA) and present[node, t] on one 15-min grid, saved as .npy for memmap. Nodes are
    streamed one at a time; the grid comes from the label cache written by scan_remote.py."""
    labels = {
        p.stem: p for r in racks for p in (cfg["long_dir"].parent / "labels" / r).glob("*.parquet")
    }
    stamps = [
        pq.read_table(p, columns=["timestamp"]).column(0).to_pandas() for p in labels.values()
    ]
    t0, t1 = min(s.min() for s in stamps), max(s.max() for s in stamps)
    grid = pd.date_range(t0, t1, freq=STEP)
    nodes = sorted(labels, key=int)
    row = {n: i for i, n in enumerate(nodes)}
    out = cache_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    shape = (len(nodes), len(grid), len(features))
    X = np.lib.format.open_memmap(out / "X.npy", "w+", np.float32, shape)
    states = np.full(shape[:2], -1, np.int8)
    present = np.zeros(shape[:2], bool)
    for rack in racks:
        with tarfile.open(cfg["raw_dir"] / f"{rack}.tar") as tar:
            for m in tar.getmembers():
                buf = io.BytesIO(tar.extractfile(m).read())
                have = [c for c in features if c in set(pq.read_schema(buf).names)]
                f = pq.read_table(buf, columns=["timestamp", "value", *have]).to_pandas()
                i, idx = (
                    row[m.name.removesuffix(".parquet")],
                    ((f["timestamp"] - t0) // STEP).to_numpy(),
                )
                wide = np.full(shape[1:], np.nan, np.float32)
                wide[idx[:, None], [features.index(c) for c in have]] = f[have].to_numpy(np.float32)
                X[i] = pd.DataFrame(wide).ffill().fillna(0).to_numpy(np.float32)
                states[i, idx] = f["value"].fillna(-1).to_numpy(np.int8)
                present[i, idx] = True
        print(f"rack {rack} loaded", flush=True)
    X.flush()
    np.save(out / "states.npy", states)
    np.save(out / "present.npy", present)
    meta = {
        "nodes": nodes,
        "t0": t0.isoformat(),
        "n": len(grid),
        "features": features,
        "racks": racks,
    }
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"cached {len(nodes)} nodes x {len(grid)} windows x {len(features)} features", flush=True)


def run(cfg, name: str, max_t: int | None) -> None:
    out = cache_dir(cfg)
    meta = json.loads((out / "meta.json").read_text())
    X = np.load(out / "X.npy", mmap_mode="r")
    states, present = np.load(out / "states.npy"), np.load(out / "present.npy")
    nodes, T = meta["nodes"], min(meta["n"], max_t or meta["n"])
    det = build(cfg["detection"]["detectors"][name], meta["features"])
    grid = pd.date_range(pd.Timestamp(meta["t0"]), periods=meta["n"], freq=STEP)

    n = int(present[:, :T].sum())  # preallocated: ~20 bytes per score instead of Python objects
    node_idx, t_idx = np.empty(n, np.int16), np.empty(n, np.int32)
    score, thr = np.full(n, np.nan), np.full(n, np.nan)
    flagged, state = np.zeros(n, bool), np.full(n, -1, np.int8)
    row = {node: i for i, node in enumerate(nodes)}
    k, start, step_t = 0, time.monotonic(), []
    for t in range(T):
        idx = np.flatnonzero(present[:, t])
        if not len(idx):
            continue
        step_t.append(t)
        block = np.asarray(X[idx, t], dtype=np.float64)
        vectors = {nodes[i]: block[j] for j, i in enumerate(idx)}
        st = {nodes[i]: (None if states[i, t] < 0 else int(states[i, t])) for i in idx}
        for node, s in det.step(vectors, st).items():
            th, flag = det.threshold.update(node, s)
            node_idx[k], t_idx[k], score[k], flagged[k] = row[node], t, s, flag
            thr[k] = np.nan if th is None else th
            state[k] = -1 if st[node] is None else st[node]
            k += 1
        if t and t % 5000 == 0:
            rate = k / (time.monotonic() - start)
            print(f"{name}: {t}/{T} windows, {rate:,.0f} vectors/s", flush=True)

    labeled = state >= 0
    table = pa.table({
        "detector": pa.array(np.full(k, name)).dictionary_encode(),
        "node_id": pa.DictionaryArray.from_arrays(node_idx[:k], pa.array(nodes)),
        "event_time": pa.array(grid[t_idx[:k]]),
        "score": pa.array(score[:k], mask=np.isnan(score[:k])),
        "threshold": pa.array(thr[:k], mask=np.isnan(thr[:k])),
        "flagged": flagged[:k],
        "nagios_state": pa.array(state[:k], mask=~labeled[:k]),
        "is_anomaly": pa.array(np.isin(state[:k], list(ANOMALY_STATES)), mask=~labeled[:k]),
    })  # fmt: skip
    (out / "scores").mkdir(exist_ok=True)
    pq.write_table(table, out / "scores" / f"{name}.parquet")
    if getattr(det, "events", None) is not None:
        events = [{**e, "time": str(grid[step_t[e["step"] - 1]])} for e in det.events]
        (out / "scores" / f"{name}.events.json").write_text(json.dumps(events, indent=1))
    print(f"{name}: done, {k} scores in {time.monotonic() - start:.0f}s", flush=True)


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--racks", default="14,4,18,1,32,45,28,15")
    p.add_argument("--load", action="store_true", help="(re)build the array cache from tarballs")
    p.add_argument("--detectors", default=",".join(cfg["detection"]["detectors"]))
    p.add_argument("--max-t", type=int, help="only the first N windows (quick benchmark)")
    args = p.parse_args()

    features = [f"{m}_{s}" for m in cfg["metrics"] for s in cfg["stats"]]
    if args.load:
        load_racks(cfg, args.racks.split(","), features)
    procs = [Process(target=run, args=(cfg, d, args.max_t)) for d in args.detectors.split(",")]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()
