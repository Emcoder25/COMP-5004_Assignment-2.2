"""Batch and incremental pipeline composition."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np

from ._validation import supervised_chunk

def _validate_steps(steps: Iterable[tuple[str, object]]) -> list[tuple[str, object]]:
    result = list(steps)
    if not result:
        raise ValueError("steps must contain at least one named component.")
    names = [name for name, _ in result]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("step names must be non-empty and unique.")
    return result


class Compose:
    """Apply named transformers sequentially."""

    def __init__(self, steps: Iterable[tuple[str, object]]) -> None:
        self.steps = _validate_steps(steps)
        self._initialized = set()

    def fit(self, X: Any, y: Any = None) -> "Compose":
        current = X
        for name, transformer in self.steps:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError(f"step {name!r} must implement fit and transform.")
            transformer.fit(current)
            self._initialized.add(name)
            current = transformer.transform(current)
        return self

    def partial_fit(self, X: Any, y: Any = None) -> "Compose":
        current = X
        for name, transformer in self.steps:
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(current)
            elif name not in self._initialized:
                transformer.fit(current)
                self._initialized.add(name)
            current = transformer.transform(current)
        return self

    def transform(self, X: Any) -> Any:
        current = X
        for _, transformer in self.steps:
            current = transformer.transform(current)
        return current

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        self.fit(X, y)
        return self.transform(X)


class FeatureUnion:
    """Fit branches on the same input and concatenate feature outputs."""

    def __init__(self, transformer_list: Iterable[tuple[str, object]]) -> None:
        self.transformer_list = _validate_steps(transformer_list)
        self._initialized: set[str] = set()

    def fit(self, X: Any, y: Any = None) -> "FeatureUnion":
        for name, transformer in self.transformer_list:
            transformer.fit(X)
            self._initialized.add(name)
        return self

    def partial_fit(self, X: Any, y: Any = None) -> "FeatureUnion":
        for name, transformer in self.transformer_list:
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(X)
            elif name not in self._initialized:
                transformer.fit(X)
                self._initialized.add(name)
        return self

    def transform(self, X: Any) -> np.ndarray:
        outputs = [np.asarray(transformer.transform(X)) for _, transformer in self.transformer_list]
        outputs = [value.reshape(-1, 1) if value.ndim == 1 else value for value in outputs]
        if any(value.ndim != 2 or value.shape[0] != outputs[0].shape[0] for value in outputs):
            raise ValueError("FeatureUnion branches must return 2D arrays with equal sample counts.")
        return np.concatenate(outputs, axis=1)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class Pipeline:
    """Sequential transformers followed by an optional supervised estimator.

    Incremental transformers are updated before transforming the chunk. A
    non-incremental transformer is fitted only on its first chunk. Estimators
    with ``partial_fit`` are never refitted through ``fit`` on later chunks.
    """

    def __init__(self, steps: Iterable[tuple[str, object]]) -> None:
        self.steps = _validate_steps(steps)
        self.named_steps = dict(self.steps)
        self._transformer_names = self.steps[:-1] if hasattr(self.steps[-1][1], "predict") else self.steps
        self._initialized: set[str] = set()
        self._is_fitted = False

    @property
    def estimator(self) -> object | None:
        final = self.steps[-1][1]
        return final if hasattr(final, "predict") else None

    @property
    def is_fitted_(self) -> bool:
        if self.estimator is not None:
            return bool(getattr(self.estimator, "is_fitted_", self._is_fitted))
        return self._is_fitted

    def _apply_fit(self, X: Any) -> Any:
        current = X
        for name, transformer in self._transformer_names:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError(f"pipeline transformer {name!r} must implement fit and transform.")
            transformer.fit(current)
            self._initialized.add(name)
            current = transformer.transform(current)
        return current

    def fit(self, X: Any, y: Any = None) -> "Pipeline":
        self._is_fitted = False
        self._initialized.clear()
        if hasattr(self, "_estimator_initialized"):
            del self._estimator_initialized
        if self.estimator is not None:
            if y is None:
                raise ValueError("y is required when the final pipeline step is an estimator.")
            X, y = supervised_chunk(X, y)
        current = self._apply_fit(X)
        estimator = self.estimator
        if estimator is not None:
            if not hasattr(estimator, "fit"):
                raise TypeError("final estimator must implement fit.")
            estimator.fit(current, y)
            self._estimator_initialized = True
        self._is_fitted = True
        return self

    def partial_fit(self, X: Any, y: Any = None) -> "Pipeline":
        if self.estimator is not None:
            if y is None:
                raise ValueError("y is required when the final pipeline step is an estimator.")
            X, y = supervised_chunk(X, y)
        current = X
        for name, transformer in self._transformer_names:
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(current)
            elif name not in self._initialized:
                transformer.fit(current)
                self._initialized.add(name)
            current = transformer.transform(current)
        estimator = self.estimator
        if estimator is not None:
            if hasattr(estimator, "partial_fit"):
                estimator.partial_fit(current, y)
            elif not hasattr(self, "_estimator_initialized"):
                estimator.fit(current, y)
                self._estimator_initialized = True
            else:
                raise TypeError("final estimator has no partial_fit; refusing repeated batch fit.")
        self._is_fitted = True
        return self

    def transform(self, X: Any) -> Any:
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform.")
        current = X
        for _, transformer in self._transformer_names:
            current = transformer.transform(current)
        return current

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: Any) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before predict.")
        if self.estimator is None:
            raise AttributeError("Pipeline final step does not implement predict.")
        return np.asarray(self.estimator.predict(self.transform(X)))

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self._is_fitted or self.estimator is None:
            raise RuntimeError("Pipeline must be fitted with a final estimator.")
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Pipeline final estimator does not implement predict_proba.")
        return np.asarray(self.estimator.predict_proba(self.transform(X)))

    def score(self, X: Any, y: Any) -> float:
        targets = np.asarray(y)
        predictions = self.predict(X)
        if targets.shape != predictions.shape:
            raise ValueError(f"y shape {targets.shape} does not match prediction shape {predictions.shape}.")
        return float(np.mean(targets == predictions))
