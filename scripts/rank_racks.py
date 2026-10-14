"""Rank racks and time windows by independent node-specific anomaly events."""

import argparse

import numpy as np
import pandas as pd

from producer.config import ROOT
from scripts.scan_anomalies import CLUSTER_WIDE_FRACTION

STEP = pd.Timedelta("15min")


def load_matrix() -> tuple[np.ndarray, list[str], list[str], pd.DatetimeIndex]:
    """labels[node, t] on a common 15-min grid: 0/2/3, -1 for NA or missing."""
    paths = sorted((ROOT / "data" / "labels").glob("*/*.parquet"))
    frames = {p: pd.read_parquet(p) for p in paths}
    t0 = min(f["timestamp"].min() for f in frames.values())
    t1 = max(f["timestamp"].max() for f in frames.values())
    grid = pd.date_range(t0, t1, freq=STEP)
    labels = np.full((len(paths), len(grid)), -1, dtype=np.int8)
    for i, f in enumerate(frames.values()):
        idx = ((f["timestamp"] - t0) // STEP).to_numpy()
        labels[i, idx] = f["value"].fillna(-1).to_numpy(dtype=np.int8)
    return labels, [p.stem for p in paths], [p.parent.name for p in paths], grid


def onsets(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Node-specific critical windows and the onsets of their episodes."""
    crit = labels == 2
    reporting = (labels >= 0).sum(axis=0)
    share = np.divide(
        crit.sum(axis=0), reporting, out=np.zeros(labels.shape[1]), where=reporting > 0
    )
    ns = crit & (share < CLUSTER_WIDE_FRACTION)
    prev_normal = np.zeros_like(ns)
    prev_normal[:, 1:] = labels[:, :-1] == 0
    return ns, ns & prev_normal


def episodes(ns: np.ndarray, onset: np.ndarray) -> pd.DataFrame:
    """One row per node-specific episode: node index, onset index, length in windows."""
    rows = []
    for i, t in zip(*np.nonzero(onset), strict=True):
        end = t
        while end < ns.shape[1] and ns[i, end]:
            end += 1
        rows.append((i, t, end - t))
    return pd.DataFrame(rows, columns=["node", "t", "length"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window-days", type=int, default=28)
    p.add_argument("--racks", type=int, default=3, help="racks to pick per window")
    p.add_argument("--min-len", type=int, default=4, help="episode windows to count as event")
    args = p.parse_args()

    labels, nodes, racks, grid = load_matrix()
    ns, onset = onsets(labels)
    eps = episodes(ns, onset)
    eps["rack"] = np.array(racks)[eps["node"]]
    rack_ids = sorted(set(racks), key=int)
    nodes_per_rack = pd.Series(racks).value_counts()
    long = eps[eps["length"] >= args.min_len]

    table = pd.DataFrame({
        "rack": rack_ids,
        "nodes": [nodes_per_rack[r] for r in rack_ids],
        "ns_episodes": [int((eps["rack"] == r).sum()) for r in rack_ids],
        "events": [eps.loc[eps["rack"] == r, "t"].nunique() for r in rack_ids],
        f"events >= {args.min_len} windows": [
            long.loc[long["rack"] == r, "t"].nunique() for r in rack_ids
        ],
        "median_len": [eps.loc[eps["rack"] == r, "length"].median() for r in rack_ids],
    })  # fmt: skip
    table["median_len"] = table["median_len"].fillna(0).map(lambda v: f"{v:g}")
    table = table.sort_values(f"events >= {args.min_len} windows", ascending=False)

    # best windows: per day-aligned window, the racks with most independent events (>= min_len)
    win, warm = args.window_days * 96, 192
    event_t = {r: np.unique(long.loc[long["rack"] == r, "t"]) for r in rack_ids}
    best = []
    for s in range(0, labels.shape[1] - win, 96):
        counts = {r: int(((t >= s + warm) & (t < s + win)).sum()) for r, t in event_t.items()}
        top = sorted(counts, key=counts.get, reverse=True)[: args.racks]
        best.append((grid[s], sum(counts[r] for r in top), top, [counts[r] for r in top]))
    best.sort(key=lambda b: -b[1])
    picked = []
    for b in best:
        if all(abs((b[0] - q[0]).days) >= args.window_days for q in picked):
            picked.append(b)
        if len(picked) == 8:
            break

    md = (
        "# Rack ranking (all racks, full 2020-2022 history)\n\n"
        f"Node-specific = Nagios state 2 while < {CLUSTER_WIDE_FRACTION:.0%} of reporting nodes "
        "are critical. Episode = run of node-specific critical windows after a normal window. "
        "Events = distinct onset timestamps within a rack (simultaneous onsets on several nodes "
        f"of one rack count once); the `>= {args.min_len}` column ignores shorter flickers.\n\n"
        + table.to_markdown(index=False)
        + f"\n\n## Best {args.window_days}-day windows for {args.racks} racks\n\n"
        f"Events of >= {args.min_len} windows, counted after a 2-day warm-up.\n\n"
        "| start | independent events | racks (events each) |\n|---|---|---|\n"
        + "\n".join(
            f"| {b[0]:%Y-%m-%d} | {b[1]} | "
            + ", ".join(f"{r} ({c})" for r, c in zip(b[2], b[3], strict=True))
            + " |"
            for b in picked
        )
        + "\n"
    )
    out = ROOT / "docs" / "data_selection" / "rack_ranking.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
