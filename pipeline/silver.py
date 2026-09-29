"""Silver layer: typing, data-quality rules, de-duplication, partitioned Parquet."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SCHEMA = ("trip_id LONG, pickup_ts TIMESTAMP, pu_zone INT, do_zone INT, passengers INT, "
          "distance_mi DOUBLE, duration_min DOUBLE, fare_usd DOUBLE, payment STRING")

def rules() -> dict:
    """Data-quality rules, in priority order (built lazily: needs an active SparkContext)."""
    return {
        "missing_passengers": F.col("passengers").isNull(),
        "negative_fare": F.col("fare_usd") <= 0,
        "impossible_distance": (F.col("distance_mi") <= 0) | (F.col("distance_mi") > 100),
        "impossible_duration": (F.col("duration_min") <= 0) | (F.col("duration_min") > 300),
    }


def apply_quality(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Return (clean rows, rejected rows with the first failed rule)."""
    reason = F.lit(None).cast("string")
    for name, cond in reversed(list(rules().items())):
        reason = F.when(cond, F.lit(name)).otherwise(reason)
    tagged = df.withColumn("dq_reason", reason)
    clean = tagged.where(F.col("dq_reason").isNull()).drop("dq_reason")
    rejected = tagged.where(F.col("dq_reason").isNotNull())
    return clean, rejected


def deduplicate(df: DataFrame) -> DataFrame:
    return df.dropDuplicates(["trip_id"])


def enrich(df: DataFrame) -> DataFrame:
    return (df.withColumn("pickup_date", F.to_date("pickup_ts"))
              .withColumn("pickup_hour", F.hour("pickup_ts"))
              .withColumn("weekday", F.date_format("pickup_ts", "E"))
              .withColumn("speed_mph", F.round(F.col("distance_mi") / (F.col("duration_min") / 60), 2)))


def build_silver(spark, lake: str) -> dict:
    raw = spark.read.schema(SCHEMA).json(f"{lake}/bronze/trips")
    n_raw = raw.count()
    clean, rejected = apply_quality(raw)
    clean = enrich(deduplicate(clean))
    (clean.repartition("pickup_date").write.mode("overwrite")
          .partitionBy("pickup_date").parquet(f"{lake}/silver/trips"))
    rejected.write.mode("overwrite").parquet(f"{lake}/quarantine/trips")
    report = {r["dq_reason"]: r["count"] for r in rejected.groupBy("dq_reason").count().collect()}
    n_clean = spark.read.parquet(f"{lake}/silver/trips").count()
    report.update({"raw": n_raw, "clean": n_clean,
                   "duplicates": n_raw - sum(v for k, v in report.items()) - n_clean})
    return report
