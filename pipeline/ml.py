"""Trip-duration prediction with Spark MLlib (pipeline + GBT regressor)."""
from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import functions as F


def train_duration_model(spark, lake: str, sample: float = 0.2, seed: int = 42):
    df = (spark.read.parquet(f"{lake}/silver/trips").sample(fraction=sample, seed=seed)
               .withColumn("is_weekend", F.col("weekday").isin("Sat", "Sun").cast("int")))
    train, test = df.randomSplit([0.8, 0.2], seed=seed)
    pipe = Pipeline(stages=[
        StringIndexer(inputCol="weekday", outputCol="weekday_idx"),
        OneHotEncoder(inputCols=["weekday_idx"], outputCols=["weekday_oh"]),
        VectorAssembler(inputCols=["distance_mi", "pickup_hour", "is_weekend", "pu_zone", "weekday_oh"],
                        outputCol="features"),
        GBTRegressor(labelCol="duration_min", featuresCol="features", maxIter=40, maxDepth=5, seed=seed),
    ])
    model = pipe.fit(train)
    pred = model.transform(test)
    ev = RegressionEvaluator(labelCol="duration_min", predictionCol="prediction")
    baseline = test.agg(F.avg("duration_min")).first()[0]
    base_mae = test.select(F.avg(F.abs(F.col("duration_min") - F.lit(baseline)))).first()[0]
    metrics = {"rmse_min": round(ev.evaluate(pred, {ev.metricName: "rmse"}), 2),
               "mae_min": round(ev.evaluate(pred, {ev.metricName: "mae"}), 2),
               "r2": round(ev.evaluate(pred, {ev.metricName: "r2"}), 3),
               "baseline_mae_min": round(base_mae, 2),
               "n_train": train.count(), "n_test": test.count()}
    model.write().overwrite().save(f"{lake}/models/duration_gbt")
    return metrics, pred.select("duration_min", "prediction", "distance_mi")
