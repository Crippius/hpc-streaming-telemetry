"""Pipeline benchmark in an isolated compose stack (stop the main stack first: same ports)."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

from producer.config import ROOT, load_config
from producer.replay_producer import SCHEMA_PATH

PROJECT = "hpc-bench"
RUNS = {"max speed (full month)": (0, None), "20 windows/s (1 week)": (20, 672),
        "5 windows/s (1 week)": (5, 672)}  # fmt: skip
DETECTORS = "capymoa_hst,nagios_persistence"


def compose(docker, *args, capture=False):
    env = {**os.environ, "COMPOSE_PROJECT_NAME": PROJECT}
    return subprocess.run([docker, "compose", *args], env=env, check=True, text=True,
                          capture_output=capture).stdout  # fmt: skip


def psql(docker, sql: str) -> str:
    return compose(docker, "exec", "-T", "timescaledb", "psql", "-U", "telemetry", "-tAc", sql,
                   capture=True).strip()  # fmt: skip


def wait(what, timeout, check):
    for _ in range(timeout):
        if check():
            return
        time.sleep(1)
    sys.exit(f"timed out waiting for {what}")


def register_schema(cfg):
    """Register up front, so Spark is streaming before the first event (no startup in latency)."""
    body = json.dumps({"schema": SCHEMA_PATH.read_text()}).encode()
    url = f"{cfg['kafka']['schema_registry_url']}/subjects/{cfg['kafka']['topic']}-value/versions"
    req = urllib.request.Request(
        url, body, {"Content-Type": "application/vnd.schemaregistry.v1+json"}
    )
    wait("schema registry", 120, lambda: _try(req))


def _try(req) -> bool:
    try:
        urllib.request.urlopen(req)
        return True
    except OSError:
        return False


def percentiles(sql_expr: str, table: str, where: str, docker) -> dict:
    row = psql(
        docker,
        f"""
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {sql_expr}),
               percentile_cont(0.95) WITHIN GROUP (ORDER BY {sql_expr}),
               percentile_cont(0.99) WITHIN GROUP (ORDER BY {sql_expr})
        FROM {table} WHERE {where}""",
    )
    return dict(zip(("p50", "p95", "p99"), (float(x) for x in row.split("|")), strict=True))


def run(docker, cfg, rate, max_windows) -> dict:
    py = sys.executable
    compose(docker, "down", "-v")
    compose(
        docker,
        "up",
        "-d",
        "--build",
        "kafka",
        "kafka-init",
        "schema-registry",
        "timescaledb",
        "spark",
    )
    register_schema(cfg)
    wait("Spark query start", 300,
         lambda: "streaming " in compose(docker, "logs", "spark", capture=True))  # fmt: skip
    det_env = {**os.environ, "CAPYMOA_JVM_ARGS": "-Xmx1g -Xss10M"}
    det_cmd = [py, "-W", "ignore", "-m", "streaming.online_detector", "--detectors", DETECTORS,
               "--group", "bench", "--exit-when-idle", "30"]  # fmt: skip
    det_log = open(ROOT / "data" / "benchmark_detector.log", "a")
    detector = subprocess.Popen(det_cmd, env=det_env, stdout=det_log, stderr=subprocess.STDOUT)
    time.sleep(15)  # detector subscribed and JVM up before the first event
    args = [py, "-W", "ignore", "-m", "producer.replay_producer", "--windows-per-sec", str(rate)]
    out = subprocess.run(args + (["--max-windows", str(max_windows)] if max_windows else []),
                         check=True, text=True, capture_output=True).stdout  # fmt: skip
    done = re.search(r"done: (\d+) msgs in ([\d.]+)s \(([\d,]+) msg/s avg\)", out)
    msgs, secs = int(done[1]), float(done[2])

    windows = max_windows or 2688
    expected = len(cfg["nodes"]) * windows // 4 * len(cfg["metrics"]) * len(cfg["stats"])
    t0 = time.monotonic()
    wait(
        "all hourly aggregates",
        1800,
        lambda: (
            int(psql(docker, "SELECT count(*) FROM metric_agg_1h WHERE n_samples = 4")) >= expected
        ),
    )
    drained_after = time.monotonic() - t0
    detector.wait(timeout=3600)

    batches = pd.DataFrame(
        [(int(r), int(ms)) for r, ms in re.findall(r"batch \d+: (\d+) rows, (\d+) ms",
                                                  compose(docker, "logs", "spark", capture=True))],
        columns=["rows", "ms"],
    )  # fmt: skip
    batches = batches[batches["rows"] > 0]
    lat = percentiles(
        "extract(epoch FROM written_at - last_publish_time)", "metric_agg_1h", "true", docker
    )
    lag = {d: percentiles("extract(epoch FROM scored_at - last_publish_time)", "anomaly_scores",
                          f"detector = '{d}'", docker) for d in DETECTORS.split(",")}  # fmt: skip
    return {
        "events": msgs,
        "producer msg/s": round(msgs / secs),
        "Spark batches": len(batches),
        "rows/batch (median)": int(batches["rows"].median()),
        "batch duration ms p50 / p95": f"{batches['ms'].median():.0f} / "
        f"{np.percentile(batches['ms'], 95):.0f}",
        "end-to-end s p50 / p95 / p99": "{p50:.1f} / {p95:.1f} / {p99:.1f}".format(**lat),
        "Spark done after producer (s)": round(drained_after, 1),
        **{f"{d} lag s p50 / p95": "{p50:.2f} / {p95:.2f}".format(**v) for d, v in lag.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docker", default="docker")
    p.add_argument("--runs", default=",".join(RUNS), help="subset of run names")
    args = p.parse_args()
    cfg = load_config()
    rows = {}
    for name in args.runs.split(","):
        rows[name] = run(args.docker, cfg, *RUNS[name])
        print(name, rows[name], flush=True)
    compose(args.docker, "down", "-v")
    md = (
        "# Pipeline benchmark\n\n"
        f"June 2020 month, {len(cfg['nodes'])} nodes x 48 features (one Kafka message per "
        "value). Each run: fresh compose project (empty Kafka, checkpoint and database), single "
        "broker, Spark local[*] with 5 s micro-batches, producer and detector on the host. "
        "End-to-end = TimescaleDB write time minus the Kafka publish timestamp of the last event "
        "in the hourly window. Detector lag = scoring time minus the publish time of the last "
        f"event of the node-window, for detectors consuming live ({DETECTORS}).\n\n"
        + pd.DataFrame(rows).T.to_markdown()
        + "\n"
    )
    (ROOT / "benchmarks" / "pipeline.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
