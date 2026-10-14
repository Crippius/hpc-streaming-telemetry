"""Evaluate stored detector scores against the Nagios labels, with bootstrap CIs."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import psycopg
from sklearn.metrics import average_precision_score, roc_auc_score

from producer.config import ROOT, load_config
from scripts.scan_anomalies import CLUSTER_WIDE_FRACTION
from streaming.detectors import build

WINDOW = pd.Timedelta("15min")
MIN_EPISODE = 4  # windows; shorter critical flickers are not counted as events
N_BOOT = 1000
# threshold parameter calibrated per detector kind; model parameters are never tuned
THRESHOLD_GRID = {
    "capymoa": ("quantile", [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]),
    "river_hst": ("quantile", [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]),
    "relative_entropy": ("k", [1.5, 2, 3, 5, 8, 12]),
    "pearson_peer": ("k", [0.5, 1, 1.5, 2, 3, 4]),
}
COLUMNS = ["detector", "node_id", "event_time", "score", "flagged", "nagios_state", "is_anomaly"]


def load_db(dsn: str) -> dict[str, pd.DataFrame]:
    with psycopg.connect(dsn) as conn:
        df = pd.read_sql(f"SELECT {', '.join(COLUMNS)} FROM anomaly_scores", conn)
    return {d: g for d, g in df.groupby("detector")}


def load_offline(cfg, name: str) -> pd.DataFrame:
    path = cfg["long_dir"].parent / "offline" / "scores" / f"{name}.parquet"
    return pd.read_parquet(path, columns=COLUMNS[1:])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # plain strings first: a categorical would sort by its category order, not by value
    df = df.assign(node_id=df["node_id"].astype(str))
    df = df.sort_values(["node_id", "event_time"], ignore_index=True)
    df["node_id"] = df["node_id"].astype("category")
    df["label"] = df["is_anomaly"].map({True: 1, False: 0}).fillna(-1).astype(np.int8)
    return df.drop(columns=["is_anomaly", "nagios_state"], errors="ignore")


class Truth:
    """Everything that depends only on the labels: row kind, episodes, events, blocks."""

    NODE, CLUSTER = 1, 2  # kind codes

    def __init__(self, df: pd.DataFrame, start, val_end, end):
        lab, t = df["label"].to_numpy(), df["event_time"]
        node = df["node_id"].cat.codes.to_numpy()
        self.key_node, self.key_time = node, t.to_numpy()
        share = pd.Series(lab == 1, dtype=float).groupby(t.to_numpy()).transform("mean")
        cluster = share.to_numpy() >= CLUSTER_WIDE_FRACTION
        self.kind = np.where(lab == 1, np.where(cluster, self.CLUSTER, self.NODE), 0).astype(
            np.int8
        )
        self.val = ((t >= start) & (t < val_end) & (lab >= 0)).to_numpy()
        self.test = ((t >= val_end) & (t < end) & (lab >= 0)).to_numpy()
        self.y = lab == 1
        self.block = node.astype(np.int64) * 10_000 + (t.dt.year * 12 + t.dt.month).to_numpy()
        node_names = df["node_id"].cat.categories.to_numpy()

        same_node = np.r_[False, node[1:] == node[:-1]]
        prev = np.where(same_node, np.r_[-1, lab[:-1]], -1)
        onset = np.flatnonzero((lab == 1) & (prev == 0))
        nxt_same = np.r_[node[1:] == node[:-1], False]
        run_end = np.flatnonzero((lab == 1) & ~(np.r_[lab[1:] == 1, False] & nxt_same)) + 1
        end_idx = run_end[np.searchsorted(run_end, onset + 1)]
        eps = pd.DataFrame(
            {"s": onset, "e": end_idx, "node": node_names[node[onset]], "t": t.to_numpy()[onset]}
        )
        eps = eps[(eps["e"] - eps["s"] >= MIN_EPISODE) & self.test[eps["s"]]]
        eps["rack"] = eps["node"].astype(int) // 20
        eps["event"] = eps.groupby(["rack", "t"]).ngroup()  # simultaneous onsets in a rack
        self.episodes = eps
        self.n_events = eps["event"].nunique()


def replay_flags(df: pd.DataFrame, threshold) -> np.ndarray:
    """Re-run an online threshold over stored scores; rows are sorted by (node, time) and the
    threshold state is per node, so this is exactly the live, causal computation."""
    nodes, scores = df["node_id"].to_numpy(), df["score"].to_numpy(float)
    return np.fromiter(
        (threshold.update(n, s)[1] for n, s in zip(nodes, scores, strict=True)), bool
    )


def f1_on(mask, y, flags) -> float:
    tp = (flags & y & mask).sum()
    fp, fn = (flags & ~y & mask).sum(), (~flags & y & mask).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def ci(values: np.ndarray, point: float, fmt: str = "{:.3f}") -> str:
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return f"{fmt.format(point)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def per_node_median(y: np.ndarray, s: np.ndarray, node: np.ndarray) -> float:
    """Median of per-node ROC-AUCs. Per-node models' scores are not comparable across nodes, so
    the pooled AUC alone understates detectors whose ranking is good within each node."""
    order = np.argsort(node, kind="stable")
    y, s, node = y[order], s[order], node[order]
    bounds = np.flatnonzero(np.r_[True, node[1:] != node[:-1], True])
    aucs = [roc_auc_score(y[a:b], s[a:b]) for a, b in zip(bounds[:-1], bounds[1:], strict=True)
            if 0 < y[a:b].sum() < b - a]  # fmt: skip
    return float(np.median(aucs)) if aucs else np.nan


def metrics(df: pd.DataFrame, truth: Truth, flags: np.ndarray, rng) -> dict:
    m, y = truth.test, truth.y
    score = df["score"].to_numpy(float)[m]
    score = np.where(np.isnan(score), np.nanmin(score) - 1, score)
    c = pd.DataFrame({"b": truth.block[m], "tp": flags[m] & y[m], "fp": flags[m] & ~y[m],
                      "fn": ~flags[m] & y[m], "neg": ~y[m]}).groupby("b").sum()  # fmt: skip
    boot = c.to_numpy()[rng.integers(0, len(c), (N_BOOT, len(c)))].sum(axis=1)
    tp, fp, fn, neg = c.sum().to_numpy()
    prec = lambda a, b: a / np.maximum(a + b, 1)  # noqa: E731
    f1 = lambda a, b, d: 2 * a / np.maximum(2 * a + b + d, 1)  # noqa: E731
    node, cluster = truth.key_node[m], truth.kind[m] == Truth.CLUSTER
    res = {
        "ROC-AUC": roc_auc_score(y[m], score),
        "per-node ROC-AUC (median)": per_node_median(y[m], score, node),
        "per-node, node-specific (median)": per_node_median(
            y[m][~cluster], score[~cluster], node[~cluster]
        ),  # fmt: skip
        "PR-AUC": average_precision_score(y[m], score),
        "precision": ci(prec(boot[:, 0], boot[:, 1]), prec(tp, fp)),
        "recall": ci(prec(boot[:, 0], boot[:, 2]), prec(tp, fn)),
        "F1": ci(f1(boot[:, 0], boot[:, 1], boot[:, 2]), f1(tp, fp, fn)),
        "false alarms / node-day": ci(96 * boot[:, 1] / boot[:, 3], 96 * fp / neg, "{:.2f}"),
        "recall node-specific": flags[m & (truth.kind == Truth.NODE)].mean(),
        "recall cluster-wide": flags[m & (truth.kind == Truth.CLUSTER)].mean(),
    }
    eps = truth.episodes
    delay = np.array(
        [
            np.argmax(flags[s:e]) if flags[s:e].any() else -1
            for s, e in zip(eps["s"], eps["e"], strict=True)
        ]
    )
    ev = pd.DataFrame(
        {"event": eps["event"].to_numpy(), "delay": np.where(delay < 0, np.inf, delay)}
    )
    ev = ev.groupby("event")["delay"].min().to_numpy()
    idx = rng.integers(0, len(ev), (N_BOOT, len(ev)))
    detected, onset = np.isfinite(ev), ev == 0
    res["events detected"] = ci(detected[idx].mean(axis=1), detected.mean())
    res["onset hits"] = ci(onset[idx].mean(axis=1), onset.mean())
    res["detected within 1 h"] = ci((ev[idx] <= 4).mean(axis=1), (ev <= 4).mean())
    res["median delay (windows)"] = float(np.median(ev[detected])) if detected.any() else np.nan
    return res


_TRUTH: Truth | None = None  # set before forking workers, so it is inherited, not pickled
_FRAMES: dict[str, pd.DataFrame] | None = None


def evaluate_detector(name, spec, cfg, features) -> dict[str, dict]:
    truth, rng = _TRUTH, np.random.default_rng(0)
    df = prepare(_FRAMES[name] if _FRAMES is not None else load_offline(cfg, name))
    same = np.array_equal(df["node_id"].cat.codes.to_numpy(), truth.key_node)
    assert same and np.array_equal(df["event_time"].to_numpy(), truth.key_time), name
    out = {name: metrics(df, truth, df["flagged"].to_numpy(bool), rng)}
    if spec and spec["kind"] in THRESHOLD_GRID:
        param, grid = THRESHOLD_GRID[spec["kind"]]
        runs = {v: replay_flags(df, build({**spec, param: v}, features).threshold) for v in grid}
        val_f1 = {v: f1_on(truth.val, truth.y, f) for v, f in runs.items()}
        best = max(grid, key=val_f1.get)
        row = metrics(df, truth, runs[best], rng)
        row["validation F1"] = val_f1[best]
        out[f"{name} [{param}={best}]"] = row
    print(f"evaluated {name}", flush=True)
    return out


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["db", "offline"], default="db")
    p.add_argument("--pg-dsn", default="postgresql://telemetry:telemetry@localhost:5432/telemetry")
    p.add_argument("--val-end", default="2021-01-01T00:00:00Z", help="offline source only")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    det_cfg = cfg["detection"]["detectors"]
    features = [f"{m}_{s}" for m in cfg["metrics"] for s in cfg["stats"]]
    burn_in = cfg["detection"]["burn_in_windows"] * WINDOW
    if args.source == "db":
        frames = load_db(args.pg_dsn)
        start = pd.Timestamp(cfg["time_range"]["start"]) + burn_in
        val_end, end = (
            pd.Timestamp(cfg["detection"]["validation_end"]),
            pd.Timestamp(cfg["time_range"]["end"]),
        )
        names = sorted(frames)
    else:
        scores = cfg["long_dir"].parent / "offline" / "scores"
        # anticipation detectors predict future drains and learn from labels:
        # evaluated separately by scripts/anticipation.py
        names = sorted(
            p.stem for p in scores.glob("*.parquet") if not p.stem.startswith("anticipation")
        )
        frames = None
    raw = frames["nagios_oracle"] if frames else load_offline(cfg, "nagios_oracle")
    state3 = (raw.sort_values(["node_id", "event_time"])["nagios_state"] == 3).to_numpy()
    oracle = prepare(raw)
    del raw
    if args.source == "offline":
        start, end = oracle["event_time"].min() + burn_in, oracle["event_time"].max() + WINDOW
        val_end = pd.Timestamp(args.val_end)
    global _TRUTH, _FRAMES
    _TRUTH, _FRAMES = Truth(oracle, start, val_end, end), frames
    truth = _TRUTH

    results = {}
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
        futures = [pool.submit(evaluate_detector, n, det_cfg.get(n), cfg, features) for n in names]
        for f in futures:
            results.update(f.result())
    table = pd.DataFrame(results).T

    m = truth.test
    racks = sorted(set(oracle["node_id"].astype(int) // 20))
    kinds = truth.kind[m]
    n_state3 = int((state3 & m).sum())
    header = (
        f"# Detector comparison ({args.source})\n\n"
        f"{oracle['node_id'].nunique()} nodes (racks {racks}). "
        f"Validation {start:%Y-%m-%d} to {val_end:%Y-%m-%d} (threshold calibration only). "
        f"**All metrics are on the test period {val_end:%Y-%m-%d} to {end:%Y-%m-%d}**: "
        f"{int(m.sum())} labeled node-windows, {int(truth.y[m].sum())} anomalous "
        f"({int((kinds == Truth.NODE).sum())} node-specific, "
        f"{int((kinds == Truth.CLUSTER).sum())} cluster-wide), "
        f"**{truth.n_events} independent events** ({len(truth.episodes)} node episodes of "
        f">= {MIN_EPISODE} windows; simultaneous onsets within a rack count once). "
        f"Nagios state 3 rows in test: {n_state3}.\n\n"
        "Rows without brackets use the online thresholds from `config.yaml` (Rettig et al.'s k "
        "for relative entropy and Pearson, q = 0.95 for HST); `[param=value]` rows replay the "
        "same online threshold with the value that maximised F1 on validation. Model parameters "
        "are never tuned. `value [lo, hi]` = bootstrap 95% CI (per-window metrics: resampling "
        "node-months; event metrics: resampling events). ROC/PR-AUC use the raw score. "
        "`*_peer` = HST on node minus cluster median. Delay = windows from event onset to first "
        "flag. `nagios_persistence` = label 15 min earlier (baseline to beat); "
        "`nagios_oracle` = current label (sanity check).\n\n"
    )
    md = header + table.to_markdown(floatfmt=".3f") + "\n"
    # the stream month has too few independent events to be a benchmark: keep it out of benchmarks/
    out = (
        ROOT
        / ("benchmarks" if args.source == "offline" else "data")
        / ("results_offline.md" if args.source == "offline" else "results_stream_month.md")
    )
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
