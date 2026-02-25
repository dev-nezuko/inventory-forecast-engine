"""
Weekly Airflow sketch of the pipeline. Airflow is an optional install
(requirements-optional.txt) and this DAG is not exercised by the test suite.

The DeepAR and PyMC trainers are still stubs, so they are not wired in; the
ensemble step runs on whichever model files exist (today only the GBM one).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG

try:  # Airflow 3 ships PythonOperator in the standard provider
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.python import PythonOperator


def _module(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_ingest(**_):
    data_dir = os.environ.get("DATA_DIR", "./examples/sample_data")
    _module(
        "src.ingest.ingest",
        "--credit_csv",
        f"{data_dir}/credit.csv",
        "--panel_csv",
        f"{data_dir}/panel.csv",
    )


default_args = {
    "owner": "forecast",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "inventory_forecast_weekly",
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    default_args=default_args,
    catchup=False,
    tags=["forecast"],
) as dag:
    ingest = PythonOperator(task_id="ingest", python_callable=run_ingest)
    link = PythonOperator(
        task_id="link",
        python_callable=_module,
        op_args=["src.linking.link_panel_credit"],
    )
    rake = PythonOperator(
        task_id="rake", python_callable=_module, op_args=["src.weighting.raking"]
    )
    features = PythonOperator(
        task_id="build_features",
        python_callable=_module,
        op_args=["src.features.build_features"],
    )
    train_gbm = PythonOperator(
        task_id="train_gbm",
        python_callable=_module,
        op_args=["src.models.train_gbm_quantile", "--horizon", "12", "--holdout", "12"],
    )
    ens = PythonOperator(
        task_id="ensemble",
        python_callable=_module,
        op_args=["src.ensemble.ensemble_and_reconcile"],
    )

    ingest >> link >> rake >> features >> train_gbm >> ens
