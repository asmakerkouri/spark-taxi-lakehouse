import datetime as dt

import pytest
from pyspark.sql import SparkSession

from pipeline.generate import generate, zones
from pipeline.gold import daily_revenue, hourly_demand, top_zones
from pipeline.silver import SCHEMA, apply_quality, deduplicate, enrich


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[2]").appName("tests")
         .config("spark.sql.shuffle.partitions", "2").config("spark.ui.enabled", "false")
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    yield s
    s.stop()


def _rows(spark, rows):
    return spark.createDataFrame(rows, SCHEMA)


T = dt.datetime(2025, 1, 6, 8, 30, tzinfo=dt.UTC)


def test_quality_rules(spark):
    df = _rows(spark, [
        (1, T, 1, 2, 1, 2.0, 10.0, 12.0, "card"),       # ok
        (2, T, 1, 2, None, 2.0, 10.0, 12.0, "card"),    # missing passengers
        (3, T, 1, 2, 1, 2.0, 10.0, -5.0, "cash"),       # negative fare
        (4, T, 1, 2, 1, 999.0, 10.0, 12.0, "card"),     # impossible distance
    ])
    clean, rejected = apply_quality(df)
    assert [r.trip_id for r in clean.collect()] == [1]
    reasons = {r.trip_id: r.dq_reason for r in rejected.collect()}
    assert reasons == {2: "missing_passengers", 3: "negative_fare", 4: "impossible_distance"}


def test_dedup_and_enrich(spark):
    df = _rows(spark, [(1, T, 1, 2, 1, 3.0, 15.0, 12.0, "card")] * 3)
    out = enrich(deduplicate(df)).collect()
    assert len(out) == 1
    assert out[0].pickup_hour == 8 and out[0].weekday == "Mon" and out[0].speed_mph == 12.0


def test_generator_is_deterministic_and_dirty(spark):
    a = generate(spark, 20_000).orderBy("trip_id").limit(50).collect()
    b = generate(spark, 20_000).orderBy("trip_id").limit(50).collect()
    assert a == b
    full = generate(spark, 20_000)
    assert full.count() > 20_000                       # duplicates injected
    clean, rejected = apply_quality(full)
    assert 0 < rejected.count() < 0.05 * full.count()  # a few % dirty


def test_gold_tables(spark):
    trips = enrich(deduplicate(apply_quality(generate(spark, 20_000))[0]))
    hd = hourly_demand(trips)
    assert hd.count() == 7 * 24
    assert daily_revenue(trips).count() == 90
    tz = top_zones(trips, zones(spark), n=2).collect()
    assert len(tz) == 10 and all(r["rank"] in (1, 2) for r in tz)
