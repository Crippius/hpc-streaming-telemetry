"""Trivial baselines and diagnostics: what does the label measure, and what do models add?"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from producer.config import ROOT, load_config
from scripts.anticipation import (
    NEG_SAMPLE,
    cluster_wide_times,
    context,
    load_cache,
    node_features,
    per_node_aucs,
    summarize,
    targets,
)

IDLE_BAND = 1.10  # a window is idle when power <= 110% of the node's idle level
HORIZONS = {"1 h": 4, "6 h": 24}


def idle_level(P: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """10th percentile of each node's power over the given (training) windows."""
    return np.array(
        [np.percentile(P[i][mask[i]], 10) if mask[i].any() else np.nan for i in range(len(P))]
    )


def select_rule(X, feats, y, mask):
    """Best single feature and sign by mean per-node ROC-AUC over `mask` (training windows only)."""
    nodes = [i for i in range(len(y)) if 0 < y[i][mask[i]].sum() < mask[i].sum()]
    best = (None, 0.0, 1)
    for j, name in enumerate(feats):
        M = np.asarray(X[:, :, j], dtype=np.float32)
        a = float(np.mean([roc_auc_score(y[i][mask[i]], M[i][mask[i]]) for i in nodes]))
        for sign, auc in ((1, a), (-1, 1 - a)):
            if auc > best[1]:
                best = (name, auc, sign)
    name, _, sign = best
    sym = "+" if sign == 1 else "-"
    return f"{sym}{name}", sign * np.asarray(X[:, :, feats.index(name)], dtype=np.float32)


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[np.argmax(f1)])


def detection_row(y, score, val, test) -> dict:
    """Pooled / per-node ROC-AUC, PR-AUC, and precision, recall, F1 at the validation-F1 cut."""
    yt, st = y[test], score[test]
    per = [
        roc_auc_score(y[i][test[i]], score[i][test[i]])
        for i in range(len(y))
        if 0 < y[i][test[i]].sum() < test[i].sum()
    ]
    flagged = st >= best_threshold(y[val], score[val])
    p, r, f1, _ = precision_recall_fscore_support(yt, flagged, average="binary", zero_division=0)
    return {
        "pooled ROC-AUC": roc_auc_score(yt, st),
        "per-node median ROC-AUC": float(np.median(per)),
        "PR-AUC": average_precision_score(yt, st),
        "precision": p,
        "recall": r,
        "F1": f1,
    }


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 1) -> dict:
    """Per-node AUC of a minus b: mean difference, bootstrap CI over nodes, share of nodes won."""
    keep = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[keep]
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "difference [95% CI]": f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]",
        "nodes where model wins": f"{(d > 0).mean():.0%}",
        "nodes": len(d),
    }


def operating_ci(y, node, rule, cands, recall=0.25, n_boot=500, seed=1) -> dict:
    """Alerts per node-day at a fixed recall for the rule and each model, with a paired
    bootstrap CI over nodes of (rule - model). Rows must be grouped by node."""
    bounds = np.flatnonzero(np.r_[True, node[1:] != node[:-1], True])
    spans = list(zip(bounds[:-1], bounds[1:], strict=True))
    rng = np.random.default_rng(seed)

    def alerts(sc, yy):
        pos = sc[yy]
        k = len(pos) - int(np.ceil(recall * len(pos)))
        return 96 * float((sc >= np.partition(pos, k)[k]).mean())

    full = {"rule": alerts(rule, y), **{n: alerts(s, y) for n, s in cands.items()}}
    diffs = {n: [] for n in cands}
    for _ in range(n_boot):
        draw = rng.integers(0, len(spans), len(spans))
        idx = np.concatenate([np.arange(*spans[i]) for i in draw])
        yy, base = y[idx], alerts(rule[idx], y[idx])
        for n, s in cands.items():
            diffs[n].append(base - alerts(s[idx], yy))
    out = {}
    for n in cands:
        lo, hi = np.percentile(diffs[n], [2.5, 97.5])
        out[n] = {
            "rule": full["rule"],
            "model": full[n],
            "rule - model [95% CI]": f"{full['rule'] - full[n]:+.2f} [{lo:+.2f}, {hi:+.2f}]",
        }
    return out


def node_aucs(y: np.ndarray, score: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Per-node ROC-AUC over a (nodes x windows) mask; NaN where a node lacks a class."""
    return np.array([
        roc_auc_score(y[i][test[i]], score[i][test[i]])
        if 0 < y[i][test[i]].sum() < test[i].sum()
        else np.nan
        for i in range(len(y))
    ])  # fmt: skip


def stored_matrix(cfg, ctx, name: str) -> np.ndarray:
    """(nodes x windows) scores of a batch-mode detector; missing scores rank lowest."""
    path = cfg["long_dir"].parent / "offline" / "scores" / f"{name}.parquet"
    df = pd.read_parquet(path, columns=["node_id", "event_time", "score"])
    row = {n: i for i, n in enumerate(ctx["nodes"])}
    S = np.full(ctx["anom"].shape, np.nan, np.float32)
    t = ((df["event_time"] - ctx["grid"][0]) // pd.Timedelta("15min")).to_numpy()
    S[df["node_id"].astype(str).map(row).to_numpy(), t] = df["score"].to_numpy(np.float32)
    return np.where(np.isnan(S), np.nanmin(S) - 1, S)


def fit_gbt(ctx, t_lo, t_hi, groups, rng, seed=0):
    """GBT for the *current* label (detection ceiling), trained on windows [t_lo, t_hi)."""
    mon = ctx["valid"] & ctx["present"]
    args = (ctx["X"], ctx["cluster"], ctx["rack_med"], ctx["racks"], ctx["anom"], ctx["grid"])
    Xs, ys = [], []
    for i in range(len(ctx["nodes"])):
        t = np.flatnonzero(mon[i, t_lo:t_hi]) + t_lo
        pos, neg = t[ctx["anom"][i, t]], t[~ctx["anom"][i, t]]
        t = np.r_[pos, neg[rng.random(len(neg)) < NEG_SAMPLE]]
        if len(t):
            Xs.append(node_features(i, t, *args, groups))
            ys.append(ctx["anom"][i, t])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    model.fit(np.vstack(Xs), np.concatenate(ys))
    S = np.full(ctx["anom"].shape, np.nan, np.float32)
    for i in range(len(ctx["nodes"])):
        t = np.flatnonzero(mon[i, t_hi:]) + t_hi
        if len(t):
            S[i, t] = model.predict_proba(node_features(i, t, *args, groups))[:, 1]
    return S


def power_only_gbt(P4, fut, keep, test0, fw, rng) -> np.ndarray:
    """Anticipation GBT using only total-power statistics and their 1 h / 6 h changes."""

    def feats(i, t):
        v = P4[i]
        return np.hstack([v[t], v[t] - v[np.maximum(t - 4, 0)], v[t] - v[np.maximum(t - 24, 0)]])

    Xs, ys = [], []
    for i in range(len(P4)):
        t = np.flatnonzero(keep[i, : test0 - fw])
        pos, neg = t[fut[i, t]], t[~fut[i, t]]
        t = np.r_[pos, neg[rng.random(len(neg)) < NEG_SAMPLE]]
        Xs.append(feats(i, t))
        ys.append(fut[i, t])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    model.fit(np.vstack(Xs), np.concatenate(ys))
    S = np.full(keep.shape, np.nan, np.float32)
    for i in range(len(P4)):
        t = np.flatnonzero(keep[i, test0:]) + test0
        S[i, t] = model.predict_proba(feats(i, t))[:, 1]
    return S


def main() -> None:
    cfg = load_config()
    meta, X, states, present = load_cache(cfg)
    ctx = context(meta, X, states, present)
    nodes, grid, anom, test0 = ctx["nodes"], ctx["grid"], ctx["anom"], ctx["test0"]
    mon = ctx["valid"] & ctx["present"]
    feats = meta["features"]
    P = np.asarray(X[:, :, feats.index("total_power_avg")], dtype=np.float32)
    P4 = np.asarray(
        X[:, :, [feats.index(f"total_power_{s}") for s in ("avg", "std", "min", "max")]]
    )
    cluster_p = np.nanmedian(np.where(present, P, np.nan), axis=0)
    rng = np.random.default_rng(0)

    # ---- detection: fit on 2020-03..08, calibrate threshold on 2020-09..12, test on 2021+
    t_fit = grid.searchsorted(pd.Timestamp("2020-09-01", tz="UTC"))
    t_split = grid.searchsorted(pd.Timestamp("2021-01-01", tz="UTC"))
    val = mon.copy()
    val[:, :t_fit] = False
    val[:, t_split:] = False
    test = mon.copy()
    test[:, :t_split] = False
    before_split = mon & (np.arange(P.shape[1]) < t_split)[None, :]
    idle_det = idle_level(P, before_split)
    rule_name, R_det = select_rule(X, feats, anom, before_split)
    print("selected detection rule:", rule_name, flush=True)
    base = anom[test].mean()
    S_gbt = fit_gbt(ctx, 0, t_fit, ["telemetry", "peers"], rng)
    # fit_gbt predicts windows >= t_fit; the test mask only uses windows >= t_split
    scores = {
        f"best single feature, chosen on training windows ({rule_name})": R_det,
        "low power (-power), hand-picked": -P,
        "low power relative to the node's idle level": -(P / idle_det[:, None]),
        "power below the cluster median": -(P - cluster_p[None, :]),
        "supervised GBT, all telemetry + peers (ceiling)": S_gbt,
    }
    det = {
        "always OK (flag nothing)": {
            "pooled ROC-AUC": 0.5,
            "per-node median ROC-AUC": 0.5,
            "PR-AUC": base,
            "precision": 0.0,
            "recall": 0.0,
            "F1": 0.0,
        }
    }
    for name, s in scores.items():
        det[name] = detection_row(anom, np.nan_to_num(s, nan=-1e9), val, test)
        print("detection", name, det[name], flush=True)
    paired_rows, operating_rows = {}, {}
    rule_auc = node_aucs(anom, R_det, test)
    detection_models = {name: s for name, s in scores.items() if name.startswith("supervised")}
    for name in ["capymoa_hst", "river_hst", "capymoa_hst_peer", "river_hst_peer",
                 "relative_entropy", "pearson_peer"]:  # fmt: skip
        detection_models[f"{name} (unsupervised)"] = stored_matrix(cfg, ctx, name)
    for name, s in detection_models.items():
        paired_rows[("detection", name, "-")] = paired(node_aucs(anom, s, test), rule_auc)
    idle_share = {
        "drained windows": float(
            (
                P[test & anom]
                <= IDLE_BAND * np.broadcast_to(idle_det[:, None], P.shape)[test & anom]
            ).mean()
        ),
        "normal windows": float(
            (
                P[test & ~anom]
                <= IDLE_BAND * np.broadcast_to(idle_det[:, None], P.shape)[test & ~anom]
            ).mean()
        ),
    }

    # ---- anticipation (GRAAFE protocol, test = last 20%)
    idle_ant = idle_level(P, mon & (np.arange(P.shape[1]) < test0)[None, :])
    ant, diag = {}, {}
    for h, fw in HORIZONS.items():
        fut, keep = targets(anom, mon, fw)
        cw = cluster_wide_times(fut, keep)
        keep_tr = keep.copy()
        keep[:, :test0] = False
        node, t = np.nonzero(keep)
        y, cwt = fut[node, t], cw[t]
        sel_mask = keep_tr.copy()
        sel_mask[:, test0 - fw :] = False
        rule_h, R_h = select_rule(X, feats, fut, sel_mask)
        print(f"selected anticipation rule ({h}):", rule_h, flush=True)
        S_po = power_only_gbt(P4, fut, keep_tr, test0, fw, rng)
        cand = {
            "always OK (constant score)": np.zeros(len(y)),
            f"best single feature, chosen on training rows ({rule_h})": R_h[node, t],
            "low power (-power), hand-picked": -P[node, t],
            "low power relative to the node's idle level": -(P[node, t] / idle_ant[node]),
            "power drop over 6 h": -(P[node, t] - P[node, np.maximum(t - 24, 0)]),
            "GBT on power only (4 stats + 1 h / 6 h changes)": S_po[node, t],
        }
        for name, s in cand.items():
            ant[(name, h)] = summarize(y, s, node, cwt)
            print(
                "anticipation",
                h,
                name,
                ant[(name, h)]["per-node AUC, all onsets [95% CI]"],
                flush=True,
            )

        # within-idle / within-busy diagnostic against the saved learned models
        idle_rows = P[node, t] <= IDLE_BAND * idle_ant[node]
        models = {}
        for label, file in [
            ("GBT telemetry + peers", "telemetry_and_peers"),
            ("GBT all features", "all_features"),
        ]:
            d = np.load(ROOT / "data" / "anticipation" / f"{file}_{fw}.npz")
            assert np.array_equal(d["y"], y), "saved model scores are misaligned with the rows"
            models[label] = d["s"]
        ns = ~(y & cwt)
        rule = R_h[node, t]
        streaming = {}
        if h == "1 h":
            for pol in ["none", "periodic", "adwin", "ddm", "optwin"]:
                M = stored_matrix(cfg, ctx, f"anticipation_gbt_{pol}")
                streaming[f"streaming GBT, retrain: {pol}"] = M[node, t]
        cands = {**{f"offline {k}": v for k, v in models.items()}, **streaming}
        for label, sc in cands.items():
            both = paired(per_node_aucs(y, sc, node), per_node_aucs(y, rule, node))
            only = paired(
                per_node_aucs(y[ns], sc[ns], node[ns]), per_node_aucs(y[ns], rule[ns], node[ns])
            )
            paired_rows[(f"anticipation {h}", label, "all onsets")] = both
            paired_rows[(f"anticipation {h}", label, "node-specific onsets")] = only
        for label, row in operating_ci(y, node, rule, cands).items():
            operating_rows[(f"anticipation {h}", label)] = row
        for subset, m in [
            ("all rows", np.ones(len(y), bool)),
            ("node idle now", idle_rows),
            ("node busy now", ~idle_rows),
        ]:
            both = 0 < y[m].sum() < m.sum()
            row = {
                "rows": int(m.sum()),
                "onsets": int(y[m].sum()),
                "onset base rate": float(y[m].mean()),
                "share of onsets": float(y[m].sum() / y.sum()),
                "AUC low power": roc_auc_score(y[m], -P[node, t][m]) if both else np.nan,
            }
            for label, s in models.items():
                row[f"AUC {label}"] = (
                    roc_auc_score(y[m], s[m]) if 0 < y[m].sum() < m.sum() else np.nan
                )
            diag[(subset, h)] = row

    det_t = pd.DataFrame(det).T
    ant_cols = [
        "per-node AUC, all onsets [95% CI]",
        "per-node AUC, node-specific [95% CI]",
        "pooled AUC",
        "PR-AUC",
        "precision @ recall 25%",
        "alerts/node-day @ recall 25%",
        "alerts/node-day @ recall 50%",
    ]
    ant_t = pd.DataFrame(ant).T[ant_cols]
    diag_t = pd.DataFrame(diag).T
    paired_t = pd.DataFrame(paired_rows).T
    paired_t.index.names = ["task", "model", "onsets"]
    operating_t = pd.DataFrame(operating_rows).T
    md = (
        "# Baselines and diagnostics\n\n"
        f"{len(nodes)} nodes. Idle level = 10th percentile of a node's total power over its "
        f"training windows; a window is *idle* when power <= {IDLE_BAND:.0%} of that level.\n\n"
        "## What the label looks like in the telemetry\n\n"
        f"Detection test windows (2021-01-01 onward): {int(test.sum())} monitored, "
        f"{base:.2%} drained. "
        f"Share of windows at idle power: drained {idle_share['drained windows']:.1%}, "
        f"normal {idle_share['normal windows']:.1%}.\n\n"
        "## Detection of the current state\n\n"
        "Model fitted on 2020-03..08, threshold chosen for best F1 on 2020-09..12, all metrics on "
        "2021-01..2022-09. 'Always OK' flags nothing: ROC-AUC 0.5, PR-AUC equal to the base "
        f"rate ({base:.2%}).\n\n"
        + det_t.to_markdown(floatfmt=".3f")
        + "\n\n## Anticipation baselines (GRAAFE protocol)\n\n"
        "Same rows, split and metrics as `anticipation.md`.\n\n"
        + ant_t.to_markdown(floatfmt=".4g")
        + "\n\n## Where do the learned anticipation models add value?\n\n"
        "Pooled AUC on subsets of the test rows, by whether the node is idle at the time of "
        "prediction. `share of onsets` = fraction of all test onsets in that subset.\n\n"
        + diag_t.to_markdown(floatfmt=".3f")
        + "\n\n## Is a model better than the best single-feature rule?\n\n"
        "Paired comparison on the same nodes and rows: per-node ROC-AUC of the model minus that of "
        "the rule, averaged over nodes, with a bootstrap 95% CI over nodes (2000 resamples). A "
        "CI that excludes 0 means the model is better (or worse) than the rule beyond node-to-node "
        "noise.\n\n" + paired_t.to_markdown() + "\n\n## False alerts at the same recall\n\n"
        "Alerts per node-day needed to catch 25% of the test onsets (threshold read off the test "
        "scores), for the rule and for each model. The CI is a paired bootstrap over nodes (500 "
        "resamples) of the rule's count minus the model's; positive means the model raises fewer "
        "false alerts.\n\n" + operating_t.to_markdown(floatfmt=".2f") + "\n"
    )
    (ROOT / "benchmarks" / "baselines.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
