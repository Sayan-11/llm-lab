from __future__ import annotations

import json
from typing import Optional
from typing import Sequence
from .experiment import RunResult
import matplotlib.pyplot as plt

import pandas as pd
from IPython.display import display  # works in notebooks

from .storage import Storage


def _runs_df() -> pd.DataFrame:
    storage = Storage()
    rows = storage.fetch_runs()
    storage.close()

    data = []
    for r in rows:
        metrics = json.loads(r["summary_metrics"] or "{}")
        row = {
            "run_id": r["id"],
            "experiment": r["experiment_name"],
            "model_name": r["name"].replace(f"{r['experiment_name']}__", ""),
            "status": r["status"],
            "dataset_size": r["dataset_size"],
            "created_at": r["created_at"],
        }
        for k, v in metrics.items():
            row[k] = v
        data.append(row)

    if not data:
        return pd.DataFrame(columns=["run_id", "experiment", "status", "dataset_size", "created_at"])

    return pd.DataFrame(data)


def show_leaderboard(metric: str = "exact_match", top_n: int = 20) -> None:
    """
    Display a simple leaderboard of runs sorted by a given metric.
    Intended for usage inside notebooks.
    """
    df = _runs_df()
    if df.empty:
        print("No runs found yet.")
        return

    if metric not in df.columns:
        print(f"No metric '{metric}' found yet. Available: {list(df.columns)}")
        display(df)
        return

    df_sorted = df.sort_values(metric, ascending=False).head(top_n)
    cols = ["run_id", "experiment", "model_name", "status", "dataset_size", metric, "created_at"]
    display(df_sorted[cols])


def summarize_run(run_id: int) -> None:
    """
    Print a summary of a given run and show a small sample of trials.
    """

    storage = Storage()
    run = storage.fetch_run(run_id)
    trials = storage.fetch_trials(run_id)
    storage.close()

    if run is None:
        print(f"Run {run_id} not found.")
        return

    metrics = json.loads(run["summary_metrics"] or "{}")
    print(f"Run {run_id} – experiment: {run['experiment_name']}")
    print(f"Status: {run['status']}")
    print(f"Dataset size: {run['dataset_size']}")
    print("Summary metrics:")
    if metrics:
        for k, v in metrics.items():
            try:
                print(f"  {k}: {float(v):.4f}")
            except (TypeError, ValueError):
                print(f"  {k}: {v}")
    else:
        print("  (none)")

    # Show a small sample of trials
    rows = []
    for t in trials[:10]:
        m = json.loads(t["metrics_json"] or "{}")
        row = {
            "example_index": t["example_index"],
            "input": t["input_text"],
            "expected": t["expected_output"],
            "output": t["output_text"],
        }
        for k, v in m.items():
            row[f"metric_{k}"] = v
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        display(df)
    else:
        print("No trials logged for this run.")

def compare_models(results: Sequence[RunResult]):
    """
    Given a list of RunResult objects (e.g. from run_experiment with multiple models),
    return a DataFrame with one row per model and metrics as columns.
    Also displays the table for convenience in notebooks.
    """
    rows = []
    for r in results:
        row = {"model_name": r.model_name}
        for k, v in r.summary_metrics.items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    display(df)
    return df        

def plot_model_metric(results: Sequence[RunResult], metric: str = "exact_match"):
    """
    Plot a simple bar chart of a given metric across multiple models.
    """
    model_names = []
    values = []

    for r in results:
        if metric in r.summary_metrics:
            model_names.append(r.model_name)
            values.append(r.summary_metrics[metric])

    if not model_names:
        print(f"No metric '{metric}' found in results.")
        return

    plt.figure()
    plt.bar(model_names, values)
    plt.ylabel(metric)
    plt.xlabel("model")
    plt.title(f"{metric} by model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()