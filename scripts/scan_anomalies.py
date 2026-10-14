"""Rank nodes in the downloaded racks by Nagios anomaly labels."""

import io
import tarfile

import pandas as pd
import pyarrow.parquet as pq

from producer.config import ROOT, load_config

CLUSTER_WIDE_FRACTION = (
    0.5  # label-2 row is "cluster-wide" if >= this share of nodes has label 2 too
)


def load_labels(raw_dir) -> pd.DataFrame:
    frames = []
    for tar_path in sorted(raw_dir.glob("*.tar")):
        with tarfile.open(tar_path) as tar:
            for m in tar.getmembers():
                buf = io.BytesIO(tar.extractfile(m).read())
                df = pq.read_table(buf, columns=["timestamp", "value"]).to_pandas()
                df["timestamp"] = df["timestamp"].astype("datetime64[ns, UTC]")
                df["node_id"] = m.name.removesuffix(".parquet")
                df["rack"] = tar_path.stem
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["node_id", "timestamp"])
    n_nodes = df["node_id"].nunique()
    crit = df["value"].eq(2).fillna(False)
    nodes_crit = crit.groupby(df["timestamp"]).transform("sum")
    node_specific = crit & (nodes_crit < CLUSTER_WIDE_FRACTION * n_nodes)
    # episode = maximal run of consecutive node-specific critical windows on one node
    prev = node_specific.groupby(df["node_id"]).shift(1, fill_value=False)
    df = df.assign(
        crit=crit,
        node_specific=node_specific,
        ns_episodes=node_specific & ~prev,
        unknown=df["value"].eq(3).fillna(False),
        na=df["value"].isna(),
    )
    g = df.groupby(["node_id", "rack"])
    out = g.agg(
        rows=("timestamp", "size"),
        crit=("crit", "sum"),
        node_specific=("node_specific", "sum"),
        ns_episodes=("ns_episodes", "sum"),
        unknown=("unknown", "sum"),
        na=("na", "sum"),
    ).reset_index()
    out["cluster_wide"] = out["crit"] - out["node_specific"]
    out["crit_rate"] = out["crit"] / out["rows"]
    return out


def main() -> None:
    cfg = load_config()
    start, end = (pd.Timestamp(cfg["time_range"][k]) for k in ("start", "end"))
    df = load_labels(cfg["raw_dir"])
    overall = summarize(df)
    in_slice = summarize(df[(df["timestamp"] >= start) & (df["timestamp"] < end)])

    table = in_slice.merge(
        overall[["node_id", "crit", "crit_rate"]], on="node_id", suffixes=("", "_all")
    )
    table = table.sort_values(["ns_episodes", "node_specific"], ascending=False)

    cols = [
        "node_id",
        "rack",
        "rows",
        "crit",
        "crit_rate",
        "ns_episodes",
        "node_specific",
        "cluster_wide",
        "unknown",
        "na",
        "crit_all",
        "crit_rate_all",
    ]
    md = table[cols].to_markdown(index=False, floatfmt=".3f")
    header = (
        "# Node ranking (racks 10-12)\n\n"
        f"Slice: {start:%Y-%m-%d} to {end:%Y-%m-%d} (exclusive). `crit` = Nagios state 2, "
        "`unknown` = state 3, `na` = unmonitored. `cluster_wide` = critical rows where at least "
        f"{CLUSTER_WIDE_FRACTION:.0%} of nodes are critical at the same timestamp. "
        "`ns_episodes` = runs of consecutive node-specific critical windows. "
        "`*_all` = whole 2020-2022 history.\n\n"
    )
    out_path = ROOT / "docs" / "data_selection" / "node_ranking.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + md + "\n")

    t = table.sum(numeric_only=True)
    print(md)
    print(
        f"\n{len(table)} nodes: {t['crit']:.0f} critical rows ({t['node_specific']:.0f} "
        f"node-specific in {t['ns_episodes']:.0f} episodes, {t['cluster_wide']:.0f} cluster-wide)"
    )
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
