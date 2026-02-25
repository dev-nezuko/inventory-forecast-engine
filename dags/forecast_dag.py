from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


def _run(cmd: list[str]):
    import subprocess
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_ingest(**kwargs):
    import os
    data_dir = os.environ.get("DATA_DIR", "./examples/sample_data")
    _run(["python", "-m", "src.ingest.ingest", "--credit_csv", f"{data_dir}/credit.csv", "--panel_csv", f"{data_dir}/panel.csv"])


def run_link(**kwargs):
    _run(["python", "-m", "src.linking.link_panel_credit"])


def run_rake(**kwargs):
    _run(["python", "-m", "src.weighting.raking"])


def build_feat(**kwargs):
    _run(["python", "-m", "src.features.build_features"])


def train_lgb(**kwargs):
    _run(["python", "-m", "src.models.train_lgb_quantile", "--quantiles", "0.1", "0.5", "0.9", "--horizon", "12"])


def train_deepar(**kwargs):
    _run(["python", "-m", "src.models.train_deepar_gluonts", "--horizon", "12"])


def ensemble(**kwargs):
    _run(["python", "-m", "src.ensemble.ensemble_and_reconcile"])


default_args = {
    "owner": "forecast_mvp",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "forecast_mvp_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@weekly",
    default_args=default_args,
    catchup=False,
    tags=["mvp"],
) as dag:

    ingest = PythonOperator(task_id="ingest", python_callable=run_ingest)
    link = PythonOperator(task_id="link", python_callable=run_link)
    rake = PythonOperator(task_id="rake", python_callable=run_rake)
    features = PythonOperator(task_id="build_features", python_callable=build_feat)
    lgb = PythonOperator(task_id="train_lgb", python_callable=train_lgb)
    deepar = PythonOperator(task_id="train_deepar", python_callable=train_deepar)
    ens = PythonOperator(task_id="ensemble", python_callable=ensemble)

    ingest >> link >> rake >> features >> [lgb, deepar] >> ens
