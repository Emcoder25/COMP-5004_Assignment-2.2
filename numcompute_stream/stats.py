"""Batch and incremental descriptive statistics."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ._validation import numeric_array
from numcompute.stats import mean, median, std, minimum, maximum, quantile, percentile, histogram


def _moments(values: np.ndarray) -> tuple[float, float]:
    scale = float(np.max(np.abs(values), initial=0))
    if scale == 0:
        return 0.0, 0.0
    normalized = values / scale
    normalized_mean = float(np.mean(normalized))
    normalized_m2 = float(np.sum((normalized - normalized_mean) ** 2))
    if normalized_m2 and (
        2 * np.log(scale) + np.log(normalized_m2) > np.log(np.finfo(float).max)
    ):
        raise ValueError("X magnitude is too large to calculate stable variance.")
    mean_value = normalized_mean * scale
    m2 = normalized_m2 * scale * scale
    if not np.isfinite(mean_value) or not np.isfinite(m2):
        raise ValueError("X magnitude is too large to calculate stable moments.")
    return mean_value, m2


class StreamingStats:
    """Exact online mean/variance/min/max; optional rolling window is exact."""

    def __init__(self, window: int | None = None) -> None:
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 1):
            raise ValueError("window must be a positive integer or None.")
        self.window = window
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf
        self._values: deque[float] = deque(maxlen=window)
        self._all_values: list[float] | None = [] if window is None else None

    def update(self, X_chunk: Any) -> "StreamingStats":
        values = numeric_array(X_chunk, "X_chunk").ravel()
        if values.size == 0:
            raise ValueError("X_chunk must not be empty.")
        if self.window is not None:
            self._values.extend(values.tolist())
            sample = np.asarray(self._values, dtype=float)
            self.count = int(sample.size)
            self.mean, self.m2 = _moments(sample)
            self.minimum, self.maximum = float(np.min(sample)), float(np.max(sample))
        else:
            assert self._all_values is not None
            n = values.size
            chunk_mean, chunk_m2 = _moments(values)
            if self.count == 0:
                self.count, self.mean, self.m2 = int(n), chunk_mean, chunk_m2
            else:
                total = self.count + n
                delta = chunk_mean - self.mean
                self.mean += delta * n / total
                with np.errstate(over="ignore", invalid="ignore"):
                    self.m2 += chunk_m2 + delta * delta * self.count * n / total
                self.count = int(total)
                if not np.isfinite(self.m2) or not np.isfinite(self.mean):
                    raise ValueError("X magnitude is too large to calculate stable running moments.")
            self.minimum = min(self.minimum, float(np.min(values)))
            self.maximum = max(self.maximum, float(np.max(values)))
            self._all_values.extend(values.tolist())
        return self

    def result(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("No observations have been added.")
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.m2 / self.count,
            "std": float(np.sqrt(self.m2 / self.count)),
            "min": self.minimum,
            "max": self.maximum,
        }

    def quantile(self, q: float) -> float:
        if self.count == 0:
            raise ValueError("No observations have been added.")
        if self.window is not None:
            values = np.asarray(self._values, dtype=float)
        else:
            values = np.asarray(self._all_values, dtype=float)
        return float(np.quantile(values, q))


def update_stats(X_chunk: Any, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge a numeric chunk into a Welford state dictionary."""
    values = numeric_array(X_chunk, "X_chunk").ravel()
    if values.size == 0:
        raise ValueError("X_chunk must not be empty.")
    state = {} if state is None else state
    count = int(state.get("count", 0))
    mean_value = float(state.get("mean", 0.0))
    m2 = float(state.get("m2", 0.0))
    n = values.size
    chunk_mean, chunk_m2 = _moments(values)
    if count == 0:
        count, mean_value, m2 = int(n), chunk_mean, chunk_m2
    else:
        total = count + n
        delta = chunk_mean - mean_value
        mean_value += delta * n / total
        with np.errstate(over="ignore", invalid="ignore"):
            m2 += chunk_m2 + delta * delta * count * n / total
        count = int(total)
        if not np.isfinite(mean_value) or not np.isfinite(m2):
            raise ValueError("X magnitude is too large to calculate stable running moments.")
    state.update(
        count=count,
        mean=mean_value,
        m2=m2,
        variance=m2 / count,
        min=min(float(state.get("min", np.inf)), float(np.min(values))),
        max=max(float(state.get("max", -np.inf)), float(np.max(values))),
    )
    return state


class StreamingHistogram:
    """Exact-count expanding histogram (stores observations to preserve bins).

    Range growth recomputes counts from all retained observations, so values on
    bin edges are never reclassified due to edge drift. Memory is O(n); with a
    ``window`` it is bounded by O(window).
    """

    def __init__(self, bins: int = 10, *, window: int | None = None) -> None:
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError("bins must be a positive integer.")
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 1):
            raise ValueError("window must be a positive integer or None.")
        self.bins, self.window = bins, window
        self._values: deque[float] = deque(maxlen=window)

    def update(self, X_chunk: Any) -> "StreamingHistogram":
        values = numeric_array(X_chunk, "X_chunk").ravel()
        if values.size == 0:
            raise ValueError("X_chunk must not be empty.")
        self._values.extend(values.tolist())
        return self

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._values:
            raise ValueError("No observations have been added.")
        values = np.asarray(self._values, dtype=float)
        low, high = float(np.min(values)), float(np.max(values))
        scale = max(abs(low), abs(high), 1.0)
        scaled_values = values / scale
        low_scaled, high_scaled = low / scale, high / scale
        if low_scaled == high_scaled:
            low_scaled = float(np.nextafter(low_scaled, -np.inf))
            high_scaled = float(np.nextafter(high_scaled, np.inf))
        counts, scaled_edges = np.histogram(
            scaled_values, bins=self.bins, range=(low_scaled, high_scaled)
        )
        return counts, scaled_edges * scale


__all__ = [
    "mean", "median", "std", "minimum", "maximum", "quantile",
    "percentile", "histogram", "StreamingStats", "StreamingHistogram",
    "update_stats",
]
