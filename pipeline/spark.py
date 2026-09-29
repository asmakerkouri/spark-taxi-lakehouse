from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(app: str = "taxi-lakehouse") -> SparkSession:
    """Local mode by default; SPARK_MASTER=spark://host:7077 for a cluster."""
    return (
        SparkSession.builder.appName(app)
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", os.getenv("SHUFFLE_PARTITIONS", "16"))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", os.getenv("DRIVER_MEMORY", "3g"))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
