"""Numerically safe common helpers and parameter validation."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ._validation import positive_int


def _finite_pair(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a)
    right = np.asarray(b)
    if left.dtype.kind not in "iuf" or right.dtype.kind not in "iuf":
        raise TypeError(f"distance inputs must be real numeric arrays; received {left.dtype} and {right.dtype}.")
    if left.shape != right.shape:
        raise ValueError(f"distance inputs must have equal shapes; received {left.shape} and {right.shape}.")
    left, right = left.astype(float, copy=False), right.astype(float, copy=False)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("distance inputs must contain only finite values.")
    return left, right


def euclidean_distance(a: Any, b: Any) -> float:
    """Compute Euclidean distance using scaled differences to reduce overflow."""
    left, right = _finite_pair(a, b)
    scale = max(float(np.max(np.abs(left), initial=0)), float(np.max(np.abs(right), initial=0)))
    if scale == 0:
        return 0.0
    normalized = left.ravel() / scale - right.ravel() / scale
    with np.errstate(over="ignore"):
        result = float(np.linalg.norm(normalized) * scale)
    if not np.isfinite(result):
        raise OverflowError("Euclidean distance exceeds the floating-point range.")
    return result


def cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity with normalized vectors to avoid overflow."""
    left, right = _finite_pair(a, b)
    left, right = left.ravel(), right.ravel()
    left_scale = float(np.max(np.abs(left), initial=0))
    right_scale = float(np.max(np.abs(right), initial=0))
    if left_scale == 0 or right_scale == 0:
        return 0.0
    left_unit, right_unit = left / left_scale, right / right_scale
    denom = float(np.linalg.norm(left_unit) * np.linalg.norm(right_unit))
    return float(np.dot(left_unit, right_unit) / denom)


def manhattan_distance(a: Any, b: Any) -> float:
    """Compute Manhattan distance, rejecting an unrepresentable result."""
    left, right = _finite_pair(a, b)
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.sum(np.abs(left.ravel() - right.ravel()), dtype=np.float64))
    if not np.isfinite(result):
        raise OverflowError("Manhattan distance exceeds the floating-point range.")
    return result


def deep_sizeof(obj: Any) -> int:
    """Estimate recursively reachable Python object memory (excluding NumPy buffers).

    NumPy ``nbytes`` are included. Shared objects are counted once. This estimate
    cannot account precisely for allocator overhead or external/native buffers.
    """
    seen: set[int] = set()

    def size(value: Any) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            return total
        if isinstance(value, dict):
            return total + sum(size(k) + size(v) for k, v in value.items())
        if isinstance(value, (list, tuple, set, frozenset)):
            return total + sum(size(item) for item in value)
        if hasattr(value, "__dict__"):
            total += size(vars(value))
        return total

    return size(obj)


__all__ = ["positive_int", "euclidean_distance", "manhattan_distance", "deep_sizeof"]
