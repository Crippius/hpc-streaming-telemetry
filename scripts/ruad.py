"""RUAD replication (Molan et al., FGCS 2023): per-node LSTM autoencoder on M100."""

import argparse
import io
import json
import tarfile
from multiprocessing import get_context

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from producer.config import ROOT, load_config
from producer.etl import ANOMALY_STATES

W = 10


class RUAD(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.enc1 = nn.LSTM(n_features, 16, batch_first=True)
        self.enc2 = nn.LSTM(16, 8, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, n_features))

    def forward(self, x):
        h, _ = self.enc1(x)
        _, (last, _) = self.enc2(h)
        return self.dec(last[-1])


def load_node(cfg, rack: str, member: str) -> pd.DataFrame:
    with tarfile.open(cfg["raw_dir"] / f"{rack}.tar") as tar:
        buf = io.BytesIO(tar.extractfile(member).read())
    df = pq.read_table(buf).to_pandas().drop(columns=["__index_level_0__"], errors="ignore")
    return df.sort_values("timestamp", ignore_index=True)


def sequences(values: np.ndarray, ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Windows of W consecutive 15-min rows without time gaps; returns (X, index of last row)."""
    gap = ts.diff().dt.total_seconds().fillna(900).to_numpy() != 900
    run_start = np.maximum.accumulate(np.where(gap, np.arange(len(ts)), 0))
    ends = np.flatnonzero(np.arange(len(ts)) - run_start >= W - 1)
    idx = ends[:, None] - np.arange(W - 1, -1, -1)
    return values[idx], ends


def train_node(args) -> dict:
    """Cached per node in data/ruad/<node>.npz, so an interrupted run resumes."""
    cfg, rack, member, epochs, seed = args
    cache = ROOT / "data" / "ruad" / f"{member.removesuffix('.parquet')}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {"node": str(d["node"]), "time": d["time"], "score": d["score"], "y": d["y"]}
    res = _train_node(cfg, rack, member, epochs, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, node=res["node"], time=res["time"].astype("datetime64[ns]"),
             score=res["score"], y=res["y"])  # fmt: skip
    return res


def _train_node(cfg, rack, member, epochs, seed) -> dict:
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    df = load_node(cfg, rack, member)
    feats = df.drop(columns=["timestamp", "value"])
    split = int(len(df) * 0.8)
    train = feats.iloc[:split]
    keep_cols = train.columns[train.notna().any()]
    lo, hi = train[keep_cols].min(), train[keep_cols].max()
    scaled = ((feats[keep_cols] - lo) / (hi - lo).replace(0, 1)).ffill().fillna(0)
    X, ends = sequences(scaled.to_numpy(np.float32), df["timestamp"])
    tr = ends < split
    Xtr = torch.from_numpy(X[tr])
    n_val = len(Xtr) // 10
    model = RUAD(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, bad = np.inf, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr) - n_val)
        for b in range(0, len(perm), 256):
            batch = Xtr[perm[b : b + 256]]
            loss = (model(batch) - batch[:, -1]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = Xtr[-n_val:]
            v = (model(val) - val[:, -1]).abs().mean().item()
        best, bad = (v, 0) if v < best - 1e-5 else (best, bad + 1)
        if bad >= 2:
            break
    model.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X[~tr])
        err = torch.cat([
            (model(Xte[b : b + 4096]) - Xte[b : b + 4096, -1]).abs().sum(dim=1)
            for b in range(0, len(Xte), 4096)
        ]).numpy()  # fmt: skip
    state = df["value"].to_numpy()[ends[~tr]]
    labeled = ~pd.isna(state)
    node = member.removesuffix(".parquet")
    print(f"node {node}: {X.shape[2]} features, {len(Xtr)} train seq", flush=True)
    return {
        "node": node,
        "time": df["timestamp"].to_numpy()[ends[~tr]][labeled],
        "score": err[labeled],
        "y": np.isin(state[labeled].astype(int), list(ANOMALY_STATES)),
    }


def auc_or_nan(y, s) -> float:
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--racks", default="14,4,18,1,32,45,28,15")
    p.add_argument("--nodes-per-rack", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--compare", default="capymoa_hst,river_hst", help="offline scores to compare")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    jobs = []
    for rack in args.racks.split(","):
        with tarfile.open(cfg["raw_dir"] / f"{rack}.tar") as tar:
            names = sorted(tar.getnames())
        for m in rng.choice(names, min(args.nodes_per_rack, len(names)), replace=False):
            jobs.append((cfg, rack, m, args.epochs, 0))
    with get_context("spawn").Pool(args.workers) as pool:  # torch thread pools break with fork
        res = pool.map(train_node, jobs)

    y = np.concatenate([r["y"] for r in res])
    s = np.concatenate([r["score"] for r in res])
    per = [auc_or_nan(r["y"], r["score"]) for r in res]
    rows = {
        "RUAD (LSTM AE, W=10)": {"pooled AUC": roc_auc_score(y, s), "per-node AUC": np.nanmean(per)}
    }

    keys = pd.DataFrame({"node_id": np.concatenate([[r["node"]] * len(r["y"]) for r in res]),
                         "event_time": np.concatenate([r["time"] for r in res])})  # fmt: skip
    keys["event_time"] = pd.to_datetime(keys["event_time"], utc=True)
    for name in args.compare.split(","):
        path = cfg["long_dir"].parent / "offline" / "scores" / f"{name}.parquet"
        other = pd.read_parquet(path, columns=["node_id", "event_time", "score", "is_anomaly"])
        other["node_id"] = other["node_id"].astype(str)
        m = keys.merge(other, on=["node_id", "event_time"], how="inner").dropna()
        yy, ss = m["is_anomaly"].astype(bool).to_numpy(), m["score"].to_numpy()
        per_o = [
            auc_or_nan(g["is_anomaly"].astype(bool), g["score"]) for _, g in m.groupby("node_id")
        ]
        rows[f"{name} (same nodes and windows)"] = {
            "pooled AUC": roc_auc_score(yy, ss), "per-node AUC": np.nanmean(per_o)
        }  # fmt: skip

    md = (
        "# RUAD replication on the public M100 dataset\n\n"
        f"{len(res)} nodes ({args.nodes_per_rack} random per rack, racks {args.racks}), all IPMI "
        f"aggregates per node, chronological 80/20 split, unsupervised, W = {W}, early stopping "
        f"on the last 10% of training. {int(y.sum())} anomalous of {len(y)} labeled test "
        "windows. References: RUAD paper 0.767 (462 features incl. system metrics, Jira-cleaned "
        "labels, 0.035% anomalies, 980 nodes); dataset paper's own RUAD on this public dataset "
        "0.57 (one node).\n\n" + pd.DataFrame(rows).T.to_markdown(floatfmt=".3f") + "\n"
    )
    (ROOT / "benchmarks" / "ruad.md").write_text(md)
    (ROOT / "data" / "ruad_nodes.json").write_text(json.dumps([r["node"] for r in res]))
    print(md)


if __name__ == "__main__":
    main()
