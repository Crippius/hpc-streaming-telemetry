"""Synthetic 3-node, 1-day event cache for the CI smoke test, built with the real ETL."""

import numpy as np
import pandas as pd

from producer.config import load_config
from producer.etl import to_events


def wide_frame(cfg, node_idx: int, rng) -> pd.DataFrame:
    ts = pd.date_range(cfg["time_range"]["start"], cfg["time_range"]["end"], freq="15min",
                       inclusive="left")  # fmt: skip
    df = pd.DataFrame({"timestamp": ts})
    load = 0.5 + 0.4 * np.sin(np.arange(len(ts)) * 2 * np.pi / 96)
    for m in cfg["metrics"]:
        base = rng.uniform(20, 600) * load + rng.normal(0, 2, len(ts))
        df[f"{m}_avg"], df[f"{m}_std"] = base, np.abs(rng.normal(1, 0.2, len(ts)))
        df[f"{m}_min"], df[f"{m}_max"] = base - 2, base + 2
    state = np.zeros(len(ts), dtype=object)
    if node_idx == 1:
        state[40:48] = 2
    df["value"] = pd.array(state.astype(int), dtype="Int32")
    return df


def main() -> None:
    cfg = load_config()
    out = cfg["long_dir"]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, node in enumerate(cfg["nodes"]):
        long = to_events(wide_frame(cfg, i, rng), str(node), cfg["metrics"], cfg["stats"])
        long.to_parquet(out / f"{node}.parquet", index=False)
        print(f"{node}: {len(long)} events -> {out / f'{node}.parquet'}")


if __name__ == "__main__":
    main()
