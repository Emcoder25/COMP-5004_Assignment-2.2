"""Finite-difference gradient and Jacobian estimators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _usable_step(value: float, requested: float) -> tuple[float, float]:
    step = requested
    with np.errstate(over="ignore", invalid="ignore"):
        plus = np.float64(value) + step
        minus = np.float64(value) - step
    if plus == value and minus == value:
        step = max(requested, float(np.spacing(np.float64(value))))
        with np.errstate(over="ignore", invalid="ignore"):
            plus = np.float64(value) + step
            minus = np.float64(value) - step
    if (plus == value or not np.isfinite(plus)) and (minus == value or not np.isfinite(minus)):
        raise ValueError(f"Cannot form a representable finite-difference step at x={value!r}.")
    return float(plus), float(minus)


def _finite_scalar(value: Any) -> float:
    result = np.asarray(value)
    if result.ndim != 0:
        raise ValueError(f"finite-difference function must return a scalar; received shape {result.shape}.")
    scalar = float(result)
    if not np.isfinite(scalar):
        raise ValueError("finite-difference function returned a non-finite value.")
    return scalar


def grad(f: Callable[[Any], Any], x: Any, h: float = 1e-5, method: str = "central") -> float | np.ndarray:
    """Estimate a scalar function's gradient, dividing by actual representable steps."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite.")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")
    point = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(point)):
        raise ValueError("x must contain only finite values.")
    if point.ndim == 0:
        value = float(point)
        plus, minus = _usable_step(value, float(h))
        if method == "central":
            if np.isfinite(plus) and np.isfinite(minus):
                return (_finite_scalar(f(plus)) - _finite_scalar(f(minus))) / (plus - minus)
            neighbor = minus if np.isfinite(minus) else plus
            return (_finite_scalar(f(value)) - _finite_scalar(f(neighbor))) / (value - neighbor)
        if np.isfinite(plus):
            return (_finite_scalar(f(plus)) - _finite_scalar(f(value))) / (plus - value)
        return (_finite_scalar(f(value)) - _finite_scalar(f(minus))) / (value - minus)
    result = np.empty_like(point)
    baseline = _finite_scalar(f(point.copy())) if method == "forward" else None
    for index in np.ndindex(point.shape):
        plus, minus = _usable_step(float(point[index]), float(h))
        x_plus, x_minus = point.copy(), point.copy()
        x_plus[index], x_minus[index] = plus, minus
        if method == "central":
            if np.isfinite(plus) and np.isfinite(minus):
                result[index] = (_finite_scalar(f(x_plus)) - _finite_scalar(f(x_minus))) / (plus - minus)
            elif np.isfinite(minus):
                result[index] = (_finite_scalar(f(point)) - _finite_scalar(f(x_minus))) / (float(point[index]) - minus)
            else:
                result[index] = (_finite_scalar(f(x_plus)) - _finite_scalar(f(point))) / (plus - float(point[index]))
        else:
            if np.isfinite(plus):
                result[index] = (_finite_scalar(f(x_plus)) - baseline) / (plus - float(point[index]))
            else:
                result[index] = (baseline - _finite_scalar(f(x_minus))) / (float(point[index]) - minus)
    return result


def jacobian(F: Callable[[np.ndarray], Any], x: Any, h: float = 1e-5, method: str = "central") -> np.ndarray:
    """Estimate a vector function's Jacobian with representable finite steps."""
    point = np.asarray(x, dtype=float)
    if point.ndim == 0:
        point = point.reshape(1)
    baseline = np.asarray(F(point.copy()), dtype=float).reshape(-1)
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(baseline)):
        raise ValueError("x and function outputs must contain only finite values.")
    jac = np.empty((baseline.size, point.size), dtype=float)
    for flat_index, index in enumerate(np.ndindex(point.shape)):
        plus, minus = _usable_step(float(point[index]), float(h))
        x_plus, x_minus = point.copy(), point.copy()
        x_plus[index], x_minus[index] = plus, minus
        if method == "central":
            if np.isfinite(plus) and np.isfinite(minus):
                jac[:, flat_index] = (
                    np.asarray(F(x_plus), dtype=float).reshape(-1)
                    - np.asarray(F(x_minus), dtype=float).reshape(-1)
                ) / (plus - minus)
            elif np.isfinite(minus):
                jac[:, flat_index] = (baseline - np.asarray(F(x_minus), dtype=float).reshape(-1)) / (float(point[index]) - minus)
            else:
                jac[:, flat_index] = (np.asarray(F(x_plus), dtype=float).reshape(-1) - baseline) / (plus - float(point[index]))
        else:
            if np.isfinite(plus):
                jac[:, flat_index] = (
                    np.asarray(F(x_plus), dtype=float).reshape(-1) - baseline
                ) / (plus - float(point[index]))
            else:
                jac[:, flat_index] = (baseline - np.asarray(F(x_minus), dtype=float).reshape(-1)) / (float(point[index]) - minus)
        if not np.all(np.isfinite(jac[:, flat_index])):
            raise ValueError("finite-difference function returned a non-finite value.")
    return jac


__all__ = ["grad", "jacobian"]
