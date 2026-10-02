import hashlib
import json
import os
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    """config.yaml, or the file named by $HPC_CONFIG (used by CI)."""
    path = path or Path(os.environ.get("HPC_CONFIG", ROOT / "config.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("raw_dir", "long_dir"):
        cfg[key] = ROOT / cfg[key]
    return cfg


def melt_hash(cfg: dict) -> str:
    """Hash of the config fields that determine the melted cache."""
    keys = ("time_range", "metrics", "stats")
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def node_members(raw_dir: Path) -> dict[str, tuple[Path, str]]:
    """Map node id -> (tarball, member name) across all rack tarballs."""
    index = {}
    for tar_path in sorted(raw_dir.glob("*.tar")):
        with tarfile.open(tar_path) as tar:
            for name in tar.getnames():
                if name.endswith(".parquet"):
                    index[Path(name).stem] = (tar_path, name)
    return index
