"""Comparable timing helpers for eager functions and lazy iterators."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from platform import platform, python_version
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np


def _consume(value: Any) -> Any:
    if isinstance(value, Iterator):
        return list(value)
    return value


def benchmark(function: Callable[..., Any], *args: Any, repeats: int = 5, warmup: int = 1, consume_iterators: bool = True, **kwargs: Any) -> dict[str, Any]:
    """Run warmups and repeated timings; consume lazy iterators by default."""
    if not callable(function):
        raise TypeError("function must be callable.")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")
    run = lambda: _consume(function(*args, **kwargs)) if consume_iterators else function(*args, **kwargs)
    for _ in range(warmup):
        run()
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        run()
        timings.append(perf_counter() - start)
    return {
        "repeats": repeats,
        "minimum_seconds": min(timings),
        "median_seconds": median(timings),
        "mean_seconds": float(np.mean(timings)),
        "environment": {
            "python": python_version(),
            "numpy": np.__version__,
            "platform": platform(),
        },
        "lazy_iterators_consumed": consume_iterators,
    }


def compare_implementations(functions: dict[str, Callable[..., Any]], *args: Any, reference: str | None = None, repeats: int = 5, warmup: int = 1, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Check comparable outputs (when ``reference`` is provided), then time."""
    if not functions:
        raise ValueError("functions must contain at least one implementation.")
    if reference is not None and reference not in functions:
        raise ValueError(f"reference {reference!r} is not a named implementation.")
    if reference is not None:
        expected = _consume(functions[reference](*args, **kwargs))
        for name, function in functions.items():
            if name != reference:
                actual = _consume(function(*args, **kwargs))
                try:
                    np.testing.assert_equal(actual, expected)
                except AssertionError as error:
                    raise ValueError(f"implementation {name!r} does not match reference {reference!r}.") from error
    return {
        name: benchmark(function, *args, repeats=repeats, warmup=warmup, **kwargs)
        for name, function in functions.items()
    }


__all__ = ["benchmark", "compare_implementations"]
