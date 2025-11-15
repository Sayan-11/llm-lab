from __future__ import annotations

from typing import Any, Dict, Callable

# Signature of any metric: (example, output) -> dict[name -> float]
MetricFn = Callable[[Dict[str, Any], str], Dict[str, float]]

# Optional registry of metrics by name
METRIC_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(fn: MetricFn) -> MetricFn:
    """
    Decorator to register a custom metric.

    Usage:

        from llm_lab.metrics import register_metric

        @register_metric
        def my_metric(example, output):
            ...
            return {"my_metric": score}
    """
    METRIC_REGISTRY[fn.__name__] = fn
    return fn


@register_metric
def exact_match(example: Dict[str, Any], output: str) -> Dict[str, float]:
    """
    Simple case-insensitive exact match between expected_output and model output.

    example: a dict with key 'expected_output'
    output: model-generated string

    Returns:
        {"exact_match": 0.0 or 1.0}
    """
    expected = str(example.get("expected_output", "")).strip()
    out = str(output).strip()
    score = 1.0 if expected.lower() == out.lower() else 0.0
    return {"exact_match": score}

def resolve_metrics(names: List[str]) -> List[MetricFn]:
    """
    Given a list of metric names, return the corresponding callables.
    Raises a clear error if a name is unknown.
    """
    resolved: List[MetricFn] = []
    for name in names:
        if name not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric '{name}'. "
                f"Known metrics: {list(METRIC_REGISTRY.keys())}"
            )
        resolved.append(METRIC_REGISTRY[name])
    return resolved
