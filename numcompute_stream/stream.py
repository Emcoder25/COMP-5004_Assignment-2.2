"""Chunk-wise training and test-then-train evaluation."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable

import numpy as np

from ._validation import supervised_chunk
from .utils import deep_sizeof


class StreamTrainer:
    """Coordinate streaming updates and log prequential chunk metrics."""

    def __init__(
        self,
        model: Any = None,
        *,
        pipeline: Any = None,
        additional_metrics: dict[str, Any] | None = None,
        track_memory: bool = True,
    ) -> None:
        if model is not None and pipeline is not None:
            raise ValueError("provide either model or pipeline, not both.")
        self.model = pipeline if pipeline is not None else model
        if self.model is None:
            raise ValueError("model or pipeline must be provided.")
        self.additional_metrics = {} if additional_metrics is None else dict(additional_metrics)
        self.track_memory = track_memory
        self.logs: list[dict[str, Any]] = []
        self._total_correct = 0
        self._total_samples = 0
        self._has_trained = False

    def _trained(self) -> bool:
        if hasattr(self.model, "is_fitted_"):
            return bool(self.model.is_fitted_)
        if hasattr(self.model, "_is_fitted"):
            return bool(self.model._is_fitted)
        return self._has_trained

    def fit_chunk(self, X: Any, y: Any) -> Any:
        X, y = supervised_chunk(X, y)
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y)
        elif not self._trained() and hasattr(self.model, "fit"):
            self.model.fit(X, y)
        else:
            raise TypeError("model must support partial_fit, or fit once for its initial chunk.")
        self._has_trained = True
        return self.model

    def score_chunk(self, X: Any, y: Any) -> float:
        X, targets = supervised_chunk(X, y)
        start = perf_counter()
        predictions = np.asarray(self.model.predict(X))
        if predictions.shape != targets.shape:
            raise ValueError(f"prediction shape {predictions.shape} does not match y shape {targets.shape}.")
        chunk_correct = int(np.count_nonzero(predictions == targets))
        chunk_count = int(targets.size)
        if chunk_count == 0:
            raise ValueError("cannot score an empty chunk.")
        self._total_correct += chunk_correct
        self._total_samples += chunk_count
        row: dict[str, Any] = {
            "chunk": len(self.logs),
            "samples": chunk_count,
            "accuracy": chunk_correct / chunk_count,
            "cumulative_accuracy": self._total_correct / self._total_samples,
            "score_seconds": perf_counter() - start,
        }
        for name, metric in self.additional_metrics.items():
            if name == "auc":
                if not hasattr(self.model, "predict_proba"):
                    raise TypeError("AUC logging requires model.predict_proba().")
                probabilities = np.asarray(self.model.predict_proba(X))
                classes = getattr(self.model, "classes_", None)
                if probabilities.ndim != 2 or probabilities.shape[1] < 2:
                    raise ValueError("predict_proba must return (n_samples, n_classes) for AUC.")
                if classes is None:
                    positive_column = -1
                else:
                    matching = np.flatnonzero(classes == metric.positive_label)
                    if matching.size != 1:
                        raise ValueError(f"AUC positive label {metric.positive_label!r} is not in model.classes_.")
                    positive_column = int(matching[0])
                metric.update(targets, probabilities[:, positive_column])
            else:
                metric.update(targets, predictions)
            row[name] = metric.result()
        row["memory_bytes"] = deep_sizeof(self.model) if self.track_memory else None
        self.logs.append(row)
        return float(row["accuracy"])

    def run(self, stream: Iterable[tuple[Any, Any]]) -> list[dict[str, Any]]:
        """Prequential test-then-train processing of ``(X_chunk, y_chunk)``."""
        for X_chunk, y_chunk in stream:
            if self._trained():
                self.score_chunk(X_chunk, y_chunk)
                self.fit_chunk(X_chunk, y_chunk)
            else:
                # Cold-start chunks train the model; scoring starts once a
                # model exists because an invented baseline would be misleading.
                self.fit_chunk(X_chunk, y_chunk)
                self.logs.append(
                    {
                        "chunk": len(self.logs),
                        "samples": int(np.asarray(y_chunk).size),
                        "accuracy": None,
                        "cumulative_accuracy": self._total_correct / self._total_samples if self._total_samples else None,
                        "score_seconds": 0.0,
                        "memory_bytes": deep_sizeof(self.model) if self.track_memory else None,
                    }
                )
        return self.get_logs()

    def metric_history(self, name: str) -> np.ndarray:
        if not name:
            raise ValueError("metric name must be non-empty.")
        return np.asarray([row[name] for row in self.logs if name in row and row[name] is not None], dtype=float)

    def get_logs(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.logs]


__all__ = ["StreamTrainer"]
