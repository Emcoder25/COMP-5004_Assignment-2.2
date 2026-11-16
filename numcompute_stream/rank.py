"""Batch and incremental ranking."""

from __future__ import annotations

from typing import Any

import numpy as np

from numcompute.rank import Ranker


def rank(data: Any, method: str = "average") -> np.ndarray:
    return Ranker().rank(data, method=method)


def percentile(data: Any, q: float, interpolation: str = "linear") -> float:
    return float(Ranker().percentile(data, q, interpolation=interpolation))


class StreamingTopK:
    """Keep ordered top-k observations with stable arrival-order ties.

    Only O(k) values and source indices are retained. Equal values preserve
    arrival order. Indices refer to the full input stream.
    """

    def __init__(self, k: int, *, largest: bool = True) -> None:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer.")
        self.k, self.largest = int(k), largest
        self._values = np.empty(0, dtype=float)
        self._indices = np.empty(0, dtype=np.int64)
        self._seen = 0

    def update(self, values: Any) -> "StreamingTopK":
        chunk = np.asarray(values)
        if chunk.ndim != 1 or chunk.dtype.kind not in "iuf":
            raise TypeError(f"values must be a 1D real numeric array; received shape {chunk.shape}, dtype {chunk.dtype}.")
        chunk = chunk.astype(float, copy=False)
        if not np.all(np.isfinite(chunk)):
            raise ValueError("values must contain only finite numbers.")
        new_indices = np.arange(self._seen, self._seen + chunk.size, dtype=np.int64)
        candidates = np.concatenate((self._values, chunk))
        indices = np.concatenate((self._indices, new_indices))
        order = np.lexsort((indices, -candidates if self.largest else candidates))
        keep = order[: self.k]
        self._values, self._indices = candidates[keep], indices[keep]
        self._seen += chunk.size
        return self

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        return self._values.copy(), self._indices.copy()


class StreamingRanker:
    """Exact rank accumulator; retains all observations (O(n) memory)."""

    def __init__(self, method: str = "average") -> None:
        if method not in {"average", "dense", "ordinal"}:
            raise ValueError("method must be 'average', 'dense', or 'ordinal'.")
        self.method = method
        self._chunks: list[np.ndarray] = []

    def update(self, values: Any) -> "StreamingRanker":
        chunk = np.asarray(values)
        if chunk.ndim != 1 or chunk.dtype.kind not in "iuf":
            raise TypeError(f"values must be a 1D real numeric array; received shape {chunk.shape}, dtype {chunk.dtype}.")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("values must contain only finite numbers.")
        self._chunks.append(chunk.copy())
        return self

    def result(self) -> np.ndarray:
        if not self._chunks:
            raise ValueError("No values have been added.")
        return Ranker().rank(np.concatenate(self._chunks), method=self.method)


__all__ = ["Ranker", "rank", "percentile", "StreamingTopK", "StreamingRanker"]
