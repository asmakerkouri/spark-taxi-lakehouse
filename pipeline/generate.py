"""Distributed generation of raw taxi trips (bronze), NYC-like but synthetic.

Generated with Spark itself (spark.range), so it scales to hundreds of millions of rows
across a cluster. About 2 % of rows are deliberately dirty (nulls, negative fares,
impossible distances, duplicates) to exercise the data-quality layer.
"""
from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

N_ZONES = 60
START = "2025-01-01 00:00:00"
DAYS = 90


def zones(spark: SparkSession) -> DataFrame:
    boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    rows = [(z, f"Zone {z:02d}", boroughs[0] if z < 25 else boroughs[1 + (z % 4)]) for z in range(1, N_ZONES + 1)]
    return spark.createDataFrame(rows, "zone_id INT, zone_name STRING, borough STRING")


def generate(spark: SparkSession, n_rows: int, seed: int = 7) -> DataFrame:
    r = lambda k: F.rand(seed + k)  # noqa: E731
    base = spark.range(n_rows).withColumnRenamed("id", "trip_id")
    day = F.floor(r(1) * DAYS)
    dow = (day + 2) % 7  # 2025-01-01 is a Wednesday -> 0 = Monday ... 6 = Sunday
    is_weekend_day = dow >= 5

    def around(mu, sigma, k):  # hour drawn from a normal distribution around a peak
        return F.pmod(F.floor(F.lit(mu) + F.randn(seed + k) * sigma), F.lit(24))

    u = r(3)
    weekday_hour = (F.when(u < 0.22, around(8.3, 1.0, 20))           # morning commute
                     .when(u < 0.48, around(18.2, 1.3, 21))          # evening commute
                     .when(u < 0.90, around(13.0, 3.5, 22))          # daytime
                     .otherwise(around(23.0, 2.0, 23)))              # night
    weekend_hour = (F.when(u < 0.55, around(14.0, 3.5, 24))          # leisure, midday / afternoon
                     .when(u < 0.80, around(21.5, 2.0, 25))          # evening out
                     .otherwise(around(1.5, 1.5, 26)))               # late night
    hour = F.when(is_weekend_day, weekend_hour).otherwise(weekday_hour)
    pickup_ts = (F.unix_timestamp(F.lit(START)) + day * 86400 + hour * 3600 + F.floor(r(5) * 3600)).cast("long")
    distance = F.round(F.exp(F.randn(seed + 6) * 0.7 + 0.6), 2)  # log-normal, median ~1.8 mi
    rush = hour.isin(7, 8, 9, 17, 18, 19) & ~is_weekend_day
    speed = F.when(rush, 8 + r(7) * 6).otherwise(12 + r(7) * 10)  # mph
    duration_min = F.round(distance / speed * 60 + 2 + r(8) * 3, 1)
    fare = F.round(3.0 + 2.5 * distance + 0.5 * duration_min + F.when(rush, 2.5).otherwise(0), 2)
    df = base.select(
        "trip_id",
        F.from_unixtime(pickup_ts).cast("timestamp").alias("pickup_ts"),
        (F.floor(F.pow(r(9), 1.8) * N_ZONES) + 1).cast("int").alias("pu_zone"),
        (F.floor(r(10) * N_ZONES) + 1).cast("int").alias("do_zone"),
        F.floor(r(11) * 4 + 1).cast("int").alias("passengers"),
        distance.alias("distance_mi"),
        duration_min.alias("duration_min"),
        fare.alias("fare_usd"),
        F.when(r(12) < 0.7, "card").otherwise("cash").alias("payment"),
    )
    # inject dirty records
    dirt = r(13)
    df = (df.withColumn("fare_usd", F.when(dirt < 0.005, -F.col("fare_usd")).otherwise(F.col("fare_usd")))
            .withColumn("distance_mi", F.when((dirt >= 0.005) & (dirt < 0.01), F.lit(999.0))
                        .otherwise(F.col("distance_mi")))
            .withColumn("passengers", F.when((dirt >= 0.01) & (dirt < 0.015), F.lit(None).cast("int"))
                        .otherwise(F.col("passengers"))))
    dups = df.where(r(14) < 0.005)
    return df.unionByName(dups)


def main() -> None:
    from .spark import get_spark

    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--out", default="lake")
    a = p.parse_args()
    spark = get_spark("generate")
    generate(spark, a.rows).write.mode("overwrite").json(f"{a.out}/bronze/trips")
    zones(spark).write.mode("overwrite").option("header", True).csv(f"{a.out}/bronze/zones")
    spark.stop()


if __name__ == "__main__":
    main()
