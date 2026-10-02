"""Kafka -> 1h event-time windows per node/metric/stat -> TimescaleDB upsert."""

import json
import os
import time
import urllib.request
from functools import partial
from pathlib import Path

import psycopg
import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.avro.functions import from_avro

WINDOW = "1 hour"
WATERMARK = "30 minutes"
OUT_COLUMNS = [
    "node_id", "metric", "stat", "window_start", "window_end", "avg_value", "min_value",
    "max_value", "n_samples", "anomaly_count", "labeled_count", "last_publish_time",
]  # fmt: skip
UPSERT = f"""
INSERT INTO metric_agg_1h ({", ".join(OUT_COLUMNS)})
SELECT {", ".join(OUT_COLUMNS)} FROM stage
ON CONFLICT (node_id, metric, stat, window_start) DO UPDATE SET
    {", ".join(f"{c} = EXCLUDED.{c}" for c in OUT_COLUMNS[4:])}, written_at = now()
"""


def aggregate(events: DataFrame) -> DataFrame:
    return (
        events.withWatermark("event_time", WATERMARK)
        .groupBy(F.window("event_time", WINDOW), "node_id", "metric", "stat")
        .agg(
            F.avg("value").alias("avg_value"),
            F.min("value").alias("min_value"),
            F.max("value").alias("max_value"),
            F.count("*").alias("n_samples"),
            F.coalesce(F.sum(F.col("is_anomaly").cast("int")), F.lit(0)).alias("anomaly_count"),
            F.count("is_anomaly").alias("labeled_count"),
            F.max("publish_time").alias("last_publish_time"),
        )
        .withColumn("window_start", F.col("window.start"))
        .withColumn("window_end", F.col("window.end"))
        .select(*OUT_COLUMNS)
    )


def decode(raw: DataFrame, schema_json: str, schema_id: int) -> DataFrame:
    """Confluent wire format = magic byte 0 + 4-byte schema id + Avro body."""
    header_id = F.conv(F.hex(F.substring("value", 2, 4)), 16, 10).cast("int")
    body = F.expr("substring(value, 6, length(value) - 5)")
    checked = F.when(header_id == schema_id, body).otherwise(
        F.raise_error(F.concat(F.lit("unexpected schema id "), header_id.cast("string")))
    )
    return raw.select(
        from_avro(checked, schema_json).alias("e"), F.col("timestamp").alias("publish_time")
    ).select("e.*", "publish_time")


def fetch_schema(registry_url: str, subject: str) -> tuple[str, int]:
    url = f"{registry_url}/subjects/{subject}/versions/latest"
    while True:
        try:
            with urllib.request.urlopen(url) as r:
                body = json.load(r)
            return body["schema"], body["id"]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            print(f"waiting for schema subject {subject} (start the producer)", flush=True)
            time.sleep(5)


def write_partition(rows, dsn: str) -> None:
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE stage (LIKE metric_agg_1h INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        with cur.copy(f"COPY stage ({', '.join(OUT_COLUMNS)}) FROM STDIN") as copy:
            copy.write_row(first)
            for row in rows:
                copy.write_row(row)
        cur.execute(UPSERT)


def upsert_batch(batch: DataFrame, batch_id: int, dsn: str) -> None:
    batch.foreachPartition(partial(write_partition, dsn=dsn))


def main() -> None:
    cfg = yaml.safe_load(Path(os.environ.get("CONFIG", "config.yaml")).read_text())
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP", cfg["kafka"]["bootstrap_servers"])
    registry = os.environ.get("SCHEMA_REGISTRY_URL", cfg["kafka"]["schema_registry_url"])
    dsn = os.environ.get("PG_DSN", "postgresql://telemetry:telemetry@localhost:5432/telemetry")
    checkpoint = os.environ.get("CHECKPOINT_DIR", "checkpoints/spark_agg")
    topic = cfg["kafka"]["topic"]

    schema_json, schema_id = fetch_schema(registry, f"{topic}-value")
    spark = SparkSession.builder.appName("hpc-windowed-agg").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 500_000)
        .load()
    )
    query = (
        aggregate(decode(raw, schema_json, schema_id))
        .writeStream.outputMode("update")
        .foreachBatch(partial(upsert_batch, dsn=dsn))
        .option("checkpointLocation", checkpoint)
        .trigger(processingTime="5 seconds")
        .start()
    )
    print(f"streaming {topic} (schema id {schema_id}) -> metric_agg_1h", flush=True)
    last = -1
    while query.isActive:
        time.sleep(10)
        for p in query.recentProgress:  # every batch since the last poll, not just the newest
            if p["batchId"] > last:
                last = p["batchId"]
                print(
                    f"batch {p['batchId']}: {p['numInputRows']} rows, "
                    f"{p['durationMs'].get('triggerExecution', 0)} ms, "
                    f"watermark {p['eventTime'].get('watermark')}",
                    flush=True,
                )
    query.awaitTermination()


if __name__ == "__main__":
    main()
