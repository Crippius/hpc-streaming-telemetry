#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (3 nodes, 1 day): producer -> Kafka -> Spark ->
# TimescaleDB, and producer -> Kafka -> online detector -> TimescaleDB.
# Runs as its own compose project, so it never touches a local deployment's data.
set -euo pipefail
cd "$(dirname "$0")/.."

DOCKER=${DOCKER:-docker}
export COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME:-hpc-smoke}
export HPC_CONFIG=tests/ci/config.yaml
PY=${PYTHON:-python}
compose() { $DOCKER compose "$@"; }
psql() { compose exec -T timescaledb psql -U telemetry -tAc "$1"; }

wait_for() {  # wait_for <description> <timeout s> <command...>
  local what=$1 timeout=$2; shift 2
  for _ in $(seq "$timeout"); do "$@" && return 0; sleep 1; done
  echo "FAIL: timed out waiting for $what"; return 1
}

compose up -d --build kafka kafka-init schema-registry timescaledb spark
$PY -m scripts.make_fixture
$PY -m producer.replay_producer --windows-per-sec 0

read -r nodes metrics stats <<<"$($PY -c "
from producer.config import load_config; c = load_config()
print(len(c['nodes']), len(c['metrics']), len(c['stats']))")"
expected=$((nodes * 24 * metrics * stats))
aggregates() { [ "$(psql "SELECT count(*) FROM metric_agg_1h WHERE n_samples = 4")" -eq "$expected" ]; }
wait_for "$expected complete hourly aggregates" 600 aggregates
echo "OK: $expected hourly aggregates, all with 4 samples"

$PY -m streaming.online_detector --group smoke --exit-when-idle 15
scores=$(psql "SELECT count(*) FROM anomaly_scores WHERE detector = 'nagios_persistence'")
[ "$scores" -eq $((nodes * 96)) ] || { echo "FAIL: $scores detector scores, expected $((nodes * 96))"; exit 1; }
echo "OK: $scores detector scores"
