"""Airflow DAG: one task per lakehouse layer, daily, with retries.

Mount this repo in Airflow (e.g. /opt/project) and set LAKE_PATH / SPARK_MASTER.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = os.getenv("PROJECT_DIR", "/opt/project")
LAKE = os.getenv("LAKE_PATH", "/data/lake")
ROWS = os.getenv("ROWS", "5000000")


def layer(step: str) -> str:
    return f"cd {PROJECT} && python -m pipeline.run --lake {LAKE} --rows {ROWS} --steps {step}"


with DAG(
    dag_id="taxi_lakehouse",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["spark", "lakehouse"],
) as dag:
    bronze = BashOperator(task_id="bronze_ingest", bash_command=layer("bronze"))
    silver = BashOperator(task_id="silver_quality", bash_command=layer("silver"))
    gold = BashOperator(task_id="gold_aggregates", bash_command=layer("gold"))
    ml = BashOperator(task_id="ml_training", bash_command=layer("ml"))

    bronze >> silver >> [gold, ml]
