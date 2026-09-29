"""End-to-end run: python -m pipeline.run --rows 5000000"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .generate import generate, zones
from .gold import build_gold
from .ml import train_duration_model
from .silver import build_silver
from .spark import get_spark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--lake", default="lake")
    p.add_argument("--skip-ml", action="store_true")
    p.add_argument("--steps", default="bronze,silver,gold,ml",
                   help="comma-separated subset, used by the Airflow DAG to run one layer per task")
    a = p.parse_args()
    spark = get_spark()
    timings: dict[str, float] = {}

    def step(name, fn):
        t0 = time.perf_counter()
        result = fn()
        timings[f"{name}_s"] = round(time.perf_counter() - t0, 1)
        return result

    def bronze():
        generate(spark, a.rows).write.mode("overwrite").json(f"{a.lake}/bronze/trips")
        zones(spark).write.mode("overwrite").option("header", True).csv(f"{a.lake}/bronze/zones")

    steps = set(a.steps.split(","))
    dq = None
    if "bronze" in steps:
        step("bronze", bronze)
    if "silver" in steps:
        dq = step("silver", lambda: build_silver(spark, a.lake))
    if "gold" in steps:
        step("gold", lambda: build_gold(spark, a.lake))

    report = {"rows_requested": a.rows, "data_quality": dq, "timings": timings,
              "spark": spark.version, "cores": spark.sparkContext.defaultParallelism}
    if "ml" in steps and not a.skip_ml:
        report["ml"] = step("ml", lambda: train_duration_model(spark, a.lake)[0])
    if steps == {"bronze", "silver", "gold", "ml"} or dq is not None:
        Path(f"{a.lake}/run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
