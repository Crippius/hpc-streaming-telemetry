"""Wide per-node parquet -> long event format, cached per node in `long_dir`."""

import io
import json
import tarfile

import pandas as pd
import pyarrow.parquet as pq

from producer.config import load_config, melt_hash, node_members

ANOMALY_STATES = {2, 3}  # 2 = critical, 3 = unknown during cluster-wide incidents
LONG_COLUMNS = ["node_id", "event_time", "metric", "stat", "value", "nagios_state", "is_anomaly"]


def label_to_anomaly(state: pd.Series) -> pd.Series:
    """Nagios state -> nullable bool: 2/3 -> True, 0 -> False, NA -> NA (unmonitored)."""
    out = state.isin(ANOMALY_STATES).astype("boolean")
    out[state.isna()] = pd.NA
    return out


def check_metrics(columns, metrics: list[str], stats: list[str], node_id: str) -> None:
    available = {c.rsplit("_", 1)[0] for c in columns if c.endswith(tuple(f"_{s}" for s in stats))}
    missing = [m for m in metrics if m not in available]
    if missing:
        raise ValueError(
            f"node {node_id}: unknown metric(s) {missing}. Available: {sorted(available)}"
        )


def melt_frame(
    wide: pd.DataFrame, node_id: str, metrics: list[str], stats: list[str]
) -> pd.DataFrame:
    check_metrics(wide.columns, metrics, stats, node_id)
    value_cols = [f"{m}_{s}" for m in metrics for s in stats]
    long = wide[["timestamp", "value", *value_cols]].rename(columns={"value": "nagios_state"})
    long = long.melt(
        id_vars=["timestamp", "nagios_state"],
        value_vars=value_cols,
        var_name="col",
        value_name="value",
    )
    long[["metric", "stat"]] = long["col"].str.rsplit("_", n=1, expand=True)
    long["node_id"] = node_id
    long["event_time"] = long["timestamp"].astype("datetime64[ms, UTC]")
    long["value"] = long["value"].astype("float64")
    long["nagios_state"] = long["nagios_state"].astype("Int32")
    long["is_anomaly"] = label_to_anomaly(long["nagios_state"])
    return long[LONG_COLUMNS].sort_values(["event_time", "metric", "stat"], ignore_index=True)


def read_wide(tar_path, member: str, start, end) -> pd.DataFrame:
    with tarfile.open(tar_path) as tar:
        buf = io.BytesIO(tar.extractfile(member).read())
    df = pq.read_table(buf).to_pandas()
    ts = df["timestamp"].astype("datetime64[ns, UTC]")
    return df[(ts >= start) & (ts < end)].assign(timestamp=ts)


def main() -> None:
    cfg = load_config()
    if not cfg["nodes"]:
        raise SystemExit("config.yaml has no nodes; run scripts/scan_anomalies.py first")
    out_dir = cfg["long_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    h = melt_hash(cfg)
    start, end = (pd.Timestamp(cfg["time_range"][k]) for k in ("start", "end"))
    members = node_members(cfg["raw_dir"])

    total = 0
    for node in map(str, cfg["nodes"]):
        path = out_dir / f"{node}.parquet"
        if manifest.get(node) == h and path.exists():
            n = pq.ParquetFile(path).metadata.num_rows
            print(f"{node}: cached ({n} events)")
        else:
            wide = read_wide(*members[node], start, end)
            long = melt_frame(wide, node, cfg["metrics"], cfg["stats"])
            long.to_parquet(path, index=False)
            manifest[node] = h
            manifest_path.write_text(json.dumps(manifest, indent=2))
            n = len(long)
            print(f"{node}: {len(wide)} windows -> {n} events")
        total += n
    print(f"total: {total} events across {len(cfg['nodes'])} nodes")


if __name__ == "__main__":
    main()
