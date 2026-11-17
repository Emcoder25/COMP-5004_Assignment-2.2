from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class StreamTrainer:
    """Simple streaming trainer for incremental model/pipeline learning."""

    def __init__(self, model: Any | None = None, pipeline: Any | None = None) -> None:
        self.model = model
        self.pipeline = pipeline
        self.history: list[dict[str, float | int]] = []
        self.memory_usage: list[float] = []

    def _get_estimator(self) -> Any:
        if self.pipeline is not None:
            return self.pipeline
        if self.model is not None:
            return self.model
        raise ValueError("A model or pipeline is required for streaming training.")

    def _predict(self, X: Any) -> Any:
        estimator = self._get_estimator()
        if hasattr(estimator, "predict"):
            return estimator.predict(X)
        raise AttributeError("The configured estimator does not implement predict().")

    def fit_chunk(self, X: Any, y: Any) -> Any:
        estimator = self._get_estimator()
        if hasattr(estimator, "partial_fit"):
            estimator.partial_fit(X, y)
        elif hasattr(estimator, "fit"):
            estimator.fit(X, y)
        else:
            raise TypeError("The configured estimator does not implement fit or partial_fit.")
        return estimator

    def score_chunk(self, X: Any, y: Any) -> float:
        y_true = np.asarray(y)
        y_pred = np.asarray(self._predict(X))
        if y_true.shape != y_pred.shape:
            raise ValueError("Predictions and targets must have the same shape.")
        accuracy = float(np.mean(y_true == y_pred))
        self.history.append({"chunk": len(self.history), "accuracy": accuracy})
        self.memory_usage.append(float(np.asarray(X).nbytes / (1024 * 1024)))
        return accuracy

    def cumulative_accuracy(self) -> float:
        if not self.history:
            return 0.0
        return float(np.mean([entry["accuracy"] for entry in self.history]))

    def get_logs(self) -> list[dict[str, float | int]]:
        return list(self.history)


__all__ = ["StreamTrainer"]
