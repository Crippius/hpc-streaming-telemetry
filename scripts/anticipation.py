"""Supervised drain-onset anticipation, evaluated with GRAAFE's protocol (Molan et al. 2024)."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from producer.config import ROOT, load_config

HORIZONS = {"1 h": 4, "6 h": 24}
NEG_SAMPLE = 0.05  # training keeps all positives and this share of negatives
RECALLS = (0.25, 0.5)
N_BOOT = 1000


def load_cache(cfg):
    out = cfg["long_dir"].parent / "offline"
    meta = json.loads((out / "meta.json").read_text())
    X = np.load(out / "X.npy", mmap_mode="r")
    states, present = np.load(out / "states.npy"), np.load(out / "present.npy")
    return meta, X, states, present


def targets(anom: np.ndarray, valid: np.ndarray, fw: int) -> tuple[np.ndarray, np.ndarray]:
    """fut[n, t] = anomalous within (t, t+fw]; keep = monitored and previous fut is 0."""
    fut = np.zeros_like(anom)
    for k in range(1, fw + 1):
        fut[:, :-k] |= anom[:, k:]
    prev = np.zeros_like(fut)
    prev[:, 1:] = fut[:, :-1]
    return fut, valid & ~prev


def medians(X, racks: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, dict]:
    """Cluster median and per-rack median of every feature at every window."""
    T = X.shape[1]
    cluster = np.empty((T, X.shape[2]), np.float32)
    rack = {r: np.empty((T, X.shape[2]), np.float32) for r in np.unique(racks)}
    for s in range(0, T, chunk):
        block = np.asarray(X[:, s : s + chunk])
        cluster[s : s + chunk] = np.median(block, axis=0)
        for r in rack:
            rack[r][s : s + chunk] = np.median(block[racks == r], axis=0)
    return cluster, rack


def node_features(i, t_idx, X, cluster, rack_med, racks, anom, grid, groups) -> np.ndarray:
    """Causal features for node i at windows t_idx."""
    x = np.asarray(X[i])
    cols = []
    if "telemetry" in groups:
        lag = lambda k: x[np.maximum(t_idx - k, 0)]  # noqa: E731
        cols += [x[t_idx], x[t_idx] - lag(4), x[t_idx] - lag(24)]
    if "peers" in groups:
        cols += [x[t_idx] - cluster[t_idx], x[t_idx] - rack_med[racks[i]][t_idx]]
    if "context" in groups:
        same_rack = racks == racks[i]
        drained = anom[same_rack][:, t_idx].sum(axis=0) - anom[i, t_idx]
        cols.append(drained[:, None].astype(np.float32))
    if "calendar" in groups:
        ts = grid[t_idx]
        cols.append(np.c_[ts.hour + ts.minute / 60, ts.dayofweek].astype(np.float32))
    return np.hstack(cols)


def per_node_aucs(y: np.ndarray, s: np.ndarray, node: np.ndarray) -> np.ndarray:
    return np.array([
        roc_auc_score(y[node == n], s[node == n])
        for n in np.unique(node)
        if 0 < y[node == n].sum() < (node == n).sum()
    ])  # fmt: skip


def operating_points(y: np.ndarray, s: np.ndarray) -> dict:
    """Precision and alerts per node-day at the thresholds that catch 25% / 50% of test onsets."""
    out, pos = {}, np.sort(s[y])[::-1]
    for r in RECALLS:
        flagged = s >= pos[int(np.ceil(r * len(pos))) - 1]
        out[f"precision @ recall {r:.0%}"] = y[flagged].mean()
        out[f"alerts/node-day @ recall {r:.0%}"] = 96 * flagged.sum() / len(s)
    return out


def context(meta, X, states, present) -> dict:
    racks = np.array([int(n) // 20 for n in meta["nodes"]])
    cluster, rack_med = medians(X, racks)
    return {
        "nodes": meta["nodes"], "racks": racks, "X": X, "present": present,
        "grid": pd.date_range(pd.Timestamp(meta["t0"]), periods=meta["n"], freq="15min"),
        "anom": np.isin(states, (2, 3)), "valid": states >= 0,
        "test0": int(meta["n"] * 0.8), "cluster": cluster, "rack_med": rack_med,
    }  # fmt: skip


def run(ctx, name, groups, fw, rng) -> tuple[dict, dict]:
    nodes, racks, grid, anom, X = ctx["nodes"], ctx["racks"], ctx["grid"], ctx["anom"], ctx["X"]
    cluster, rack_med, test0 = ctx["cluster"], ctx["rack_med"], ctx["test0"]
    fut, keep = targets(anom, ctx["valid"] & ctx["present"], fw)

    tr_X, tr_y, te_s, te_y, te_node = [], [], [], [], []
    for i in range(len(nodes)):
        train_t = np.flatnonzero(keep[i, : test0 - fw])  # label must be resolved before test
        pos, neg = train_t[fut[i, train_t]], train_t[~fut[i, train_t]]
        neg = neg[rng.random(len(neg)) < NEG_SAMPLE]
        t = np.r_[pos, neg]
        tr_X.append(node_features(i, t, X, cluster, rack_med, racks, anom, grid, groups))
        tr_y.append(fut[i, t])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    model.fit(np.vstack(tr_X), np.concatenate(tr_y))
    for i in range(len(nodes)):
        t = np.flatnonzero(keep[i, test0:]) + test0
        te_s.append(
            model.predict_proba(
                node_features(i, t, X, cluster, rack_med, racks, anom, grid, groups)
            )[:, 1]
        )
        te_y.append(fut[i, t])
        te_node.append(np.full(len(t), i))
    s, y, node = np.concatenate(te_s), np.concatenate(te_y), np.concatenate(te_node)
    out = ROOT / "data" / "anticipation"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name.replace(' ', '_').replace('+', 'and')}_{fw}.npz", s=s, y=y, node=node)
    t = np.concatenate([np.flatnonzero(keep[i, test0:]) + test0 for i in range(len(nodes))])
    cw = cluster_wide_times(fut, keep)[t]
    when = grid[t]
    months = (when.tz_convert(None) if when.tz is not None else when).to_period("M")
    monthly = {}
    for m in months.unique():
        sel = np.asarray(months == m)
        ys = y[sel]
        monthly[str(m)] = {
            "onsets": int(ys.sum()),
            "cluster-wide onsets": int((ys & cw[sel]).sum()),
            "nodes with onsets": int(len(np.unique(node[sel][ys]))),
            "pooled AUC": roc_auc_score(ys, s[sel]) if 0 < ys.sum() < len(ys) else np.nan,
        }
    return summarize(y, s, node, cw), monthly


def cluster_wide_times(fut: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Windows where at least half of the eligible nodes have an onset entering the horizon."""
    return (fut & keep).sum(axis=0) >= 0.5 * np.maximum(keep.sum(axis=0), 1)


def stored_scores(ctx, cfg, name: str, fw: int, rng) -> dict:
    """Same protocol for a detector's stored batch-mode scores (data/offline/scores/<name>)."""
    df = pd.read_parquet(
        cfg["long_dir"].parent / "offline" / "scores" / f"{name}.parquet",
        columns=["node_id", "event_time", "score"],
    )
    row = {n: i for i, n in enumerate(ctx["nodes"])}
    S = np.full(ctx["anom"].shape, np.nan)
    t = ((df["event_time"] - ctx["grid"][0]) // pd.Timedelta("15min")).to_numpy()
    S[df["node_id"].astype(str).map(row).to_numpy(), t] = df["score"].to_numpy(float)
    fut, keep = targets(ctx["anom"], ctx["valid"] & ctx["present"], fw)
    cw = cluster_wide_times(fut, keep)
    keep[:, : ctx["test0"]] = False
    node, t = np.nonzero(keep)
    s = S[node, t]
    res = summarize(fut[node, t], np.where(np.isnan(s), np.nanmin(s), s), node, cw[t])
    events = cfg["long_dir"].parent / "offline" / "scores" / f"{name}.events.json"
    if events.exists():
        ev = json.loads(events.read_text())
        res["retrains"] = sum(e["reason"] != "initial" for e in ev)
    return res


def auc_ci(y, s, node, rng) -> str:
    aucs = per_node_aucs(y, s, node)
    boot = aucs[rng.integers(0, len(aucs), (N_BOOT, len(aucs)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f"{aucs.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def summarize(y, s, node, cluster_wide, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    ns = ~(y & cluster_wide)
    return {
        "per-node AUC, all onsets [95% CI]": auc_ci(y, s, node, rng),
        "per-node AUC, node-specific [95% CI]": auc_ci(y[ns], s[ns], node[ns], rng),
        "pooled AUC": roc_auc_score(y, s),
        "PR-AUC": average_precision_score(y, s),
        "onset base rate": y.mean(),
        **operating_points(y, s),
        "test onsets": int(y.sum()),
        "node-specific onsets": int((y & ns).sum()),
        "cluster-wide share of onsets": 1 - (y & ns).sum() / max(y.sum(), 1),
    }


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--ablations", action="store_true", help="also run feature-group ablations")
    p.add_argument("--scores", help="evaluate stored online-detector scores instead (1 h)")
    args = p.parse_args()
    meta, X, states, present = load_cache(cfg)
    ctx = context(meta, X, states, present)
    rng = np.random.default_rng(0)
    if args.scores:
        rows = {n: stored_scores(ctx, cfg, n, HORIZONS["1 h"], rng) for n in args.scores.split(",")}
        md = (
            "# Streaming drain-onset anticipation (GRAAFE protocol, 1 h)\n\n"
            "Detectors learn in the stream from delayed Nagios labels over the whole history; "
            "metrics on the same test period, rows and protocol as benchmarks/anticipation.md. "
            "`anticipation_{arf,hat,capymoa}` = incremental trees; `anticipation_gbt_*` = gradient "
            "boosting refit on all resolved examples by policy (none / every 4 weeks / when a "
            "CapyMOA drift detector fires on the class-balanced prequential error, at most "
            "weekly). Node-specific = onsets outside cluster-wide events (>= 50% of nodes).\n\n"
            + pd.DataFrame(rows).T.to_markdown(floatfmt=".4g")
            + "\n"
        )
        (ROOT / "benchmarks" / "anticipation_online.md").write_text(md)
        print(md)
        return
    variants = {"all features": ["telemetry", "peers", "context", "calendar"]}
    if args.ablations:
        variants |= {
            "telemetry only": ["telemetry"],
            "telemetry + peers": ["telemetry", "peers"],
            "calendar only": ["calendar"],
            "context only": ["context"],
        }
    rows, monthly = {}, {}
    for name, groups in variants.items():
        for h, fw in HORIZONS.items():
            rows[(name, h)], monthly[(name, h)] = run(ctx, name, groups, fw, rng)
            print(name, h, rows[(name, h)], flush=True)
    table = pd.DataFrame(rows).T
    head = monthly[("all features", "1 h")]
    by_month = pd.DataFrame({
        "1 h onsets": {m: v["onsets"] for m, v in head.items()},
        "of which cluster-wide": {m: v["cluster-wide onsets"] for m, v in head.items()},
        "nodes with onsets": {m: v["nodes with onsets"] for m, v in head.items()},
        **{f"AUC {n}, {h}": {m: v["pooled AUC"] for m, v in monthly[(n, h)].items()}
           for n in ("telemetry + peers", "all features") for h in HORIZONS},
    }).sort_index()  # fmt: skip
    share = rows[("all features", "1 h")]["cluster-wide share of onsets"]
    graafe = (
        "GRAAFE (same data, all 49 racks, 1 h / 6 h): rack GNN 0.91 / 0.78, "
        "per-node DNN 0.64 / 0.62, "
        "gradient boosting 0.63 / 0.60, random forest 0.61 / 0.55, majority class 0.50."
    )
    md = (
        "# Drain-onset anticipation (GRAAFE protocol)\n\n"
        f"{len(meta['nodes'])} nodes (racks {sorted({int(n) // 20 for n in meta['nodes']})}), "
        "chronological 80/20 split, rising-edge filter, AUC averaged over nodes with a test "
        f"onset; CI = bootstrap over nodes ({N_BOOT} resamples). HistGradientBoosting, all "
        f"positives + {NEG_SAMPLE:.0%} of negatives for training.\n\n"
        "Operating points are read off the test precision-recall curve (descriptive, not a "
        "calibrated threshold). A random scorer has precision = base rate and flags "
        "24 / 48 alerts per node-day for 25% / 50% recall.\n\n"
        f"{graafe}\n\n"
        f"Cluster-wide = at least 50% of the eligible nodes have an onset entering the horizon "
        f"at the same window; {share:.0%} of the 1 h test onsets are of this kind (planned drains "
        "of whole racks). `node-specific` columns drop those onset rows and keep all negatives.\n\n"
        + table.to_markdown(floatfmt=".4g")
        + "\n\n## Test period by month (pooled AUC)\n\n"
        "Months differ sharply, and cluster-wide drains are not uniformly easy or hard to "
        "anticipate, so the headline depends on which events fall in the test period.\n\n"
        + by_month.to_markdown(floatfmt=".3f")
        + "\n"
    )
    (ROOT / "benchmarks" / "anticipation.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
