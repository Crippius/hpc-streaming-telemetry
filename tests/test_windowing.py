from datetime import UTC, datetime, timedelta

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from streaming.spark_windowed_agg import aggregate

SCHEMA = StructType(
    [
        StructField("node_id", StringType()),
        StructField("event_time", TimestampType()),
        StructField("metric", StringType()),
        StructField("stat", StringType()),
        StructField("value", DoubleType()),
        StructField("is_anomaly", BooleanType()),
        StructField("publish_time", TimestampType()),
    ]
)
T0 = datetime(2020, 6, 14, tzinfo=UTC)


def event(minutes: int, value: float) -> tuple:
    t = T0 + timedelta(minutes=minutes)
    return ("215", t, "total_power", "avg", value, minutes == 60, t)


@pytest.fixture(scope="module")
def spark():
    s = (
        SparkSession.builder.master("local[2]")
        .config("spark.sql.shuffle.partitions", 2)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", False)
        .getOrCreate()
    )
    yield s
    s.stop()


def test_watermark_updates_open_window_and_drops_closed_one(spark, tmp_path):
    src, table = tmp_path / "src", {}

    def upsert(batch, _id):
        for r in batch.withColumn("key", F.date_format("window_start", "HH:mm")).collect():
            table[r["key"]] = r

    def write(name, rows):
        spark.createDataFrame(rows, SCHEMA).write.mode("append").parquet(str(src / name))

    # batch 1: 00:00..02:00 every 15 min, but 00:30 and 01:45 are held back (arrive late)
    write("b1", [event(m, float(m)) for m in range(0, 121, 15) if m not in (30, 105)])
    query = (
        aggregate(spark.readStream.schema(SCHEMA).parquet(str(src / "*")))
        .writeStream.outputMode("update")
        .foreachBatch(upsert)
        .option("checkpointLocation", str(tmp_path / "ckpt"))
        .start()
    )
    query.processAllAvailable()

    # batch 2: 01:45 belongs to [01:00, 02:00), still open (end 02:00 > watermark 01:30)
    #          00:30 belongs to [00:00, 01:00), already closed (end 01:00 <= watermark)
    write("b2", [event(105, 105.0), event(30, 30.0)])
    query.processAllAvailable()
    # a no-data batch applying the new watermark may or may not have run yet: scan all progress
    progress = query.recentProgress
    query.stop()
    watermarks = {p["eventTime"].get("watermark") for p in progress}
    dropped = sum(p["stateOperators"][0]["numRowsDroppedByWatermark"] for p in progress)

    assert "2020-06-14T01:30:00.000Z" in watermarks

    w0, w1 = table["00:00"], table["01:00"]
    assert w1["n_samples"] == 4 and w1["max_value"] == 105.0  # late-but-in-watermark: counted
    assert w1["anomaly_count"] == 1 and w1["labeled_count"] == 4
    assert w0["n_samples"] == 3 and w0["min_value"] == 0.0  # beyond watermark: dropped
    assert dropped == 1
