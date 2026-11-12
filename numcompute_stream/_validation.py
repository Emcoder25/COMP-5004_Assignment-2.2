"""Shared input validation for NumCompute-Stream."""

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np


def positive_int(value: Any, name: str) -> int:
    """Return a positive integer, rejecting booleans and non-integral values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer; got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be positive; got {value}.")
    return int(value)


def numeric_array(values: Any, name: str = "values", *, finite: bool = True) -> np.ndarray:
    """Convert to a real floating array and report received dtype on errors."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        if raw.dtype.kind == "O" and all(isinstance(item, Number) and not isinstance(item, complex) for item in raw.flat):
            result = raw.astype(np.float64, copy=False)
        else:
            raise TypeError(f"{name} must contain real numeric values; received dtype {raw.dtype}.")
    else:
        result = raw.astype(np.float64, copy=False)
    if finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values; received dtype {raw.dtype}.")
    return result


def supervised_chunk(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate a supervised chunk with samples on axis 0."""
    features = np.asarray(X)
    targets = np.asarray(y)
    if features.ndim not in (1, 2):
        raise ValueError(f"X must have shape (n_samples,) or (n_samples, n_features); received shape {features.shape}, dtype {features.dtype}.")
    if targets.ndim != 1:
        raise ValueError(f"y must have shape (n_samples,); received shape {targets.shape}, dtype {targets.dtype}.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError(f"X and y sample counts differ: X has shape {features.shape}, y has shape {targets.shape}.")
    if targets.size == 0:
        raise ValueError("X and y chunks must not be empty.")
    return features, targets
