"""Gold layer: business aggregates with Spark SQL, window functions and a broadcast join."""
from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def hourly_demand(trips: DataFrame) -> DataFrame:
    return (trips.groupBy("weekday", "pickup_hour")
                 .agg(F.count("*").alias("trips"), F.round(F.avg("fare_usd"), 2).alias("avg_fare"),
                      F.round(F.avg("speed_mph"), 1).alias("avg_speed_mph")))


def daily_revenue(trips: DataFrame) -> DataFrame:
    return (trips.groupBy("pickup_date")
                 .agg(F.count("*").alias("trips"), F.round(F.sum("fare_usd"), 0).alias("revenue_usd"))
                 .orderBy("pickup_date"))


def top_zones(trips: DataFrame, zones: DataFrame, n: int = 3) -> DataFrame:
    """Top pickup zones per borough (broadcast join of the small dimension + window rank)."""
    by_zone = trips.groupBy("pu_zone").agg(F.count("*").alias("trips"))
    joined = by_zone.join(F.broadcast(zones), by_zone.pu_zone == zones.zone_id)
    w = Window.partitionBy("borough").orderBy(F.desc("trips"))
    return (joined.withColumn("rank", F.row_number().over(w)).where(F.col("rank") <= n)
                  .select("borough", "rank", "zone_name", "trips").orderBy("borough", "rank"))


def build_gold(spark, lake: str) -> dict[str, DataFrame]:
    trips = spark.read.parquet(f"{lake}/silver/trips")
    zones = spark.read.option("header", True).schema("zone_id INT, zone_name STRING, borough STRING") \
        .csv(f"{lake}/bronze/zones")
    tables = {"hourly_demand": hourly_demand(trips), "daily_revenue": daily_revenue(trips),
              "top_zones": top_zones(trips, zones)}
    for name, df in tables.items():
        df.write.mode("overwrite").parquet(f"{lake}/gold/{name}")
    return tables
