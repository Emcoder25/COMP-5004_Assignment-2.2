"""CSV input utilities.

CSV files are expected to contain one header row followed by consistently
delimited records. ``iter_csv`` keeps only a chunk in memory. Mixed-type data
is returned as an object array; numeric columns are inferred as floating point.
Malformed records and invalid options raise ``ValueError`` or ``OSError``.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from numcompute.io import iter_csv as _iter_csv
from numcompute.io import load_csv as _load_csv
from ._validation import positive_int


def iter_csv(
    path: str | Path,
    chunksize: int = 1000,
    delimiter: str = ",",
    skip_header: int = 0,
    missing_strategy: str = "fill",
    fill_value: Any = "NaN",
    dtype: Any = None,
) -> Iterator[tuple[list[str], np.ndarray]]:
    """Yield ``(header, rows)`` chunks from a header-first CSV file."""
    positive_int(chunksize, "chunksize")
    yield from _iter_csv(path, chunksize, delimiter, skip_header, missing_strategy, fill_value, dtype)


def load_csv(
    path: str | Path,
    delimiter: str = ",",
    skip_header: int = 0,
    missing_strategy: str = "fill",
    fill_value: Any = "NaN",
    dtype: Any = None,
) -> tuple[list[str], np.ndarray]:
    """Load a header-first CSV file into one NumPy array."""
    return _load_csv(path, delimiter, skip_header, missing_strategy, fill_value, dtype)

__all__ = ["iter_csv", "load_csv"]
