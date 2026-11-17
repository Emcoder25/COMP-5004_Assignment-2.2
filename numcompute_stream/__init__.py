"""Streaming-specific NumCompute components."""

import sys

from .stream import StreamTrainer
from .tree import DecisionTree, DecisionTreeClassifier
from .visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)

__version__ = "0.2.0"

__all__ = [
    "StreamTrainer",
    "DecisionTree",
    "DecisionTreeClassifier",
    "compare_models",
    "plot_metric_over_time",
    "plot_predictions_vs_ground_truth",
]


# Compatibility aliases: expose legacy module paths through canonical
# NumCompute modules instead of maintaining duplicate implementations.
def _alias_module(legacy_name: str, canonical_name: str) -> None:
    try:
        import importlib
        sys.modules[legacy_name] = importlib.import_module(canonical_name)
    except ImportError:
        pass


_alias_module("numcompute_stream.metrics", "numcompute.metrics")
_alias_module("numcompute_stream.stats", "numcompute.stats")
_alias_module("numcompute_stream.io", "numcompute.io")
_alias_module("numcompute_stream.optim", "numcompute.optim")
_alias_module("numcompute_stream.pipeline", "numcompute.pipeline")
_alias_module("numcompute_stream.rank", "numcompute.rank")
_alias_module("numcompute_stream.sort_search", "numcompute.sort_search")
_alias_module("numcompute_stream.utils", "numcompute.utils")
_alias_module("numcompute_stream.benchmarking", "numcompute.benchmarking")

import importlib
sys.modules.setdefault("numcompute.stream", importlib.import_module("numcompute_stream.stream"))
