from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import csv

import jsonlines

from .model_client import OpenAIClient, BaseModelClient
from .metrics import exact_match, resolve_metrics
from .storage import Storage

@dataclass
class Experiment:
    """
    Definition of a simple experiment.

    - dataset_path: JSONL/JSON file with examples
    - prompt_template: Python .format template using keys from each example
    - model_name: OpenAI model id (e.g. 'gpt-4o-mini')
    """

    name: str
    dataset_path: str
    prompt_template: str
    model_names: List[str]            
    description: str = ""
    metrics: Optional[List[str]] = None
    model_params: Optional[Dict[str, Any]] = None
    provider: str = "openai"           

    def to_config(self) -> Dict[str, Any]:
        """Serializable config snapshot for reproducibility."""
        return {
             "name": self.name,
            "dataset_path": self.dataset_path,
            "prompt_template": self.prompt_template,
            "model_names": self.model_names,
            "description": self.description,
            "model_params": self.model_params or {},
            "provider": self.provider,
            "metrics": self.metrics or ["exact_match"],
        }    


@dataclass
class RunResult:
    run_id: int
    experiment_name: str
    model_name: str
    num_examples: int
    summary_metrics: Dict[str, float]

def _make_model_client(
    provider: str,
    model_name: str,
    params: Optional[Dict[str, Any]],
) -> BaseModelClient:
    """
    Factory for model clients.
    For now only 'openai' is supported.
    """
    if provider == "openai":
        return OpenAIClient(name=model_name, default_params=params)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix == ".jsonl":
        ...
    elif p.suffix == ".json":
        ...
    elif p.suffix == ".csv":
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # DictReader gives strings; keep as-is, metrics/prompt can cast.
                rows.append(row)
        return rows
    else:
        raise ValueError(f"Unsupported dataset format: {p.suffix}")


def run_experiment(
    experiment: Experiment,
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
):
    """
    Main execution loop for an experiment.

    Now supports multiple models per experiment.
    Returns:
        - RunResult if 1 model
        - List[RunResult] if multiple models
    """

    storage = Storage()
    dataset = _load_dataset(experiment.dataset_path)

    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(dataset)

    if max_examples is not None:
        dataset = dataset[: max_examples]

    dataset_size = len(dataset)

    # Register experiment once
    exp_id = storage.get_or_create_experiment(
        name=experiment.name,
        description=experiment.description,
        config=experiment.to_config(),
    )

    metric_names = experiment.metrics or ["exact_match"]
    metric_fns: List[MetricFn] = resolve_metrics(metric_names)

    all_results: List[RunResult] = []

    # Loop over each model
    for model_name in experiment.model_names:
        run_name = f"{experiment.name}__{model_name}"
        run_id = storage.create_run(exp_id, run_name, dataset_size)

        model = _make_model_client(
            provider=experiment.provider,
            model_name=model_name,
            params=experiment.model_params,
        )

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        for idx, ex in enumerate(dataset):
            # Render prompt using example fields
            try:
                prompt = experiment.prompt_template.format(**ex)
            except KeyError as e:
                raise KeyError(f"Missing key {e} in example {idx}: {ex}") from e

            # Call model
            t0 = time.time()
            try:
                output = model.generate(prompt)
            except Exception as e:
                output = f"[ERROR] {e}"
            latency_ms = (time.time() - t0) * 1000.0

            expected = str(ex.get("expected_output", ""))

            # Compute metrics
            combined_metrics: Dict[str, float] = {}
            for mfn in metric_fns:
                mres = mfn(ex, output)
                for k, v in mres.items():
                    combined_metrics[k] = float(v)
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                    metric_counts[k] = metric_counts.get(k, 0) + 1

            # Log trial
            storage.log_trial(
                run_id=run_id,
                example_index=idx,
                model_name=model_name,
                prompt_template=experiment.prompt_template,
                input_text=str(ex.get("input") or ex.get("question") or ""),
                expected_output=expected,
                output_text=output,
                metrics=combined_metrics,
                latency_ms=latency_ms,
            )

        # Aggregate summary metrics for this model
        summary: Dict[str, float] = {}
        for k, total in metric_sums.items():
            count = metric_counts.get(k, 1)
            summary[k] = total / max(count, 1)

        storage.update_run_summary(
            run_id=run_id,
            status="completed",
            summary_metrics=summary,
        )

        all_results.append(
            RunResult(
                run_id=run_id,
                experiment_name=experiment.name,
                model_name=model_name,
                num_examples=dataset_size,
                summary_metrics=summary,
            )
        )

    storage.close()

    if len(all_results) == 1:
        return all_results[0]
    return all_results
