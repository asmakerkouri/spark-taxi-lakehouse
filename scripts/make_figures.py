"""README figures from an existing lake: python scripts/make_figures.py --lake lake

Also measures the effect of partition pruning (JSON full scan vs partitioned Parquet).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from pipeline.silver import SCHEMA  # noqa: E402
from pipeline.spark import get_spark  # noqa: E402

OUT = ROOT / "docs" / "images"
SURFACE, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def style(ax, title, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)


def timed(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return round(best, 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lake", default="lake")
    lake = p.parse_args().lake
    OUT.mkdir(parents=True, exist_ok=True)
    spark = get_spark("figures")
    report = json.loads(Path(f"{lake}/run_report.json").read_text())

    # 1. demand heatmap (sequential, one hue)
    hd = spark.read.parquet(f"{lake}/gold/hourly_demand").toPandas()
    grid = hd.pivot(index="weekday", columns="pickup_hour", values="trips").reindex(DAYS)
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=130, facecolor=SURFACE)
    im = ax.imshow(grid.values / 1000, aspect="auto", cmap="Blues")
    ax.set_yticks(range(7), DAYS)
    ax.set_xticks(range(0, 24, 2), [f"{h}h" for h in range(0, 24, 2)])
    style(ax, "Trips per weekday and hour (thousands) - rush hours on weekdays", grid_axis=None)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    save(fig, "demand_heatmap.png")

    # 2. data quality
    dq = report["data_quality"]
    labels = ["missing passengers", "impossible distance", "negative fare", "duplicates", "impossible duration"]
    keys = ["missing_passengers", "impossible_distance", "negative_fare", "duplicates", "impossible_duration"]
    vals = [dq.get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 3), dpi=130, facecolor=SURFACE)
    bars = ax.barh(labels[::-1], vals[::-1], color=BLUE, height=0.55)
    for b, v in zip(bars, vals[::-1], strict=True):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", color=INK, fontsize=9)
    pct = 100 * (dq["raw"] - dq["clean"]) / dq["raw"]
    style(ax, f"Rows removed by the quality layer: {dq['raw'] - dq['clean']:,} of {dq['raw']:,} ({pct:.1f} %)",
          grid_axis="x")
    ax.set_xlim(0, max(vals) * 1.18)
    save(fig, "data_quality.png")

    # 3. partition pruning benchmark: revenue of one day
    day = "2025-02-14"
    bronze = lambda: (spark.read.schema(SCHEMA).json(f"{lake}/bronze/trips")  # noqa: E731
                      .where(F.to_date("pickup_ts") == day).agg(F.sum("fare_usd")).collect())
    silver = lambda: (spark.read.parquet(f"{lake}/silver/trips")  # noqa: E731
                      .where(F.col("pickup_date") == day).agg(F.sum("fare_usd")).collect())
    bench = {"bronze_json_full_scan_s": timed(bronze), "silver_parquet_pruned_s": timed(silver)}
    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=130, facecolor=SURFACE)
    names = ["Raw JSON (full scan)", "Parquet partitioned by date"]
    v = [bench["bronze_json_full_scan_s"], bench["silver_parquet_pruned_s"]]
    bars = ax.barh(names[::-1], v[::-1], color=BLUE, height=0.5)
    for b, x in zip(bars, v[::-1], strict=True):
        ax.text(x + max(v) * 0.01, b.get_y() + b.get_height() / 2, f"{x:.2f} s", va="center", color=INK)
    style(ax, f"Revenue of one day: {v[0] / v[1]:.0f}x faster with columnar storage + partition pruning",
          grid_axis="x")
    ax.set_xlim(0, max(v) * 1.2)
    save(fig, "partition_pruning.png")

    # 4. ML: predicted vs actual duration
    model = PipelineModel.load(f"{lake}/models/duration_gbt")
    sample = (spark.read.parquet(f"{lake}/silver/trips").sample(fraction=0.004, seed=1)
              .withColumn("is_weekend", F.col("weekday").isin("Sat", "Sun").cast("int")))
    pdf = model.transform(sample).select("duration_min", "prediction").toPandas()
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=130, facecolor=SURFACE)
    hb = ax.hexbin(pdf.duration_min, pdf.prediction, gridsize=45, cmap="Blues", mincnt=1, extent=(0, 60, 0, 60))
    ax.plot([0, 60], [0, 60], color=INK2, lw=1, ls="--")
    m = report["ml"]
    style(ax, f"GBT trip duration: R2 {m['r2']}, MAE {m['mae_min']} min", grid_axis=None)
    ax.set_xlabel("actual (min)", color=INK2)
    ax.set_ylabel("predicted (min)", color=INK2)
    fig.colorbar(hb, ax=ax, fraction=0.04).outline.set_visible(False)
    save(fig, "ml_duration.png")

    # 5. top zones table as json for the README
    tz = spark.read.parquet(f"{lake}/gold/top_zones").where("rank = 1").toPandas()
    (OUT / "benchmark.json").write_text(json.dumps({**bench, "top_zone_per_borough": tz.to_dict("records")},
                                                   indent=2, default=str))
    print(bench)
    spark.stop()


if __name__ == "__main__":
    main()
