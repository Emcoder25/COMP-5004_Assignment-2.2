"""Compare a Python loop with NumPy and exercise buffered tree sizes.

Run from the repository root with ``python benchmark/vectorized_vs_loop.py``.
Timings are demonstrations only; they vary by machine and interpreter.
"""

from __future__ import annotations

import json
import platform

import numpy as np

from numcompute_stream.benchmarking import benchmark, compare_implementations
from numcompute_stream.io import load_csv
from numcompute_stream.preprocessing import Imputer
from numcompute_stream.tree import DecisionTreeClassifier


def loop_threshold_predict(X: np.ndarray, threshold: float) -> np.ndarray:
    result = np.empty(X.shape[0], dtype=np.int64)
    for row in range(X.shape[0]):
        total = 0.0
        for column in range(X.shape[1]):
            total += X[row, column]
        result[row] = total > threshold
    return result


def vectorized_threshold_predict(X: np.ndarray, threshold: float) -> np.ndarray:
    return (X.sum(axis=1) > threshold).astype(np.int64)


def evaluate_buffer_size(
    X: np.ndarray, y: np.ndarray, buffer_size: int, chunk_size: int
) -> dict[str, float | int]:
    model = DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size, random_state=5)
    correct = 0
    scored = 0
    for start in range(0, X.shape[0], chunk_size):
        X_chunk = X[start : start + chunk_size]
        y_chunk = y[start : start + chunk_size]
        if model.is_fitted_:
            predictions = model.predict(X_chunk)
            correct += int(np.count_nonzero(predictions == y_chunk))
            scored += y_chunk.size
        model.partial_fit(X_chunk, y_chunk)
    return {
        "buffer_size": buffer_size,
        "prequential_accuracy": float(correct / scored) if scored else float("nan"),
        "retraining_count": model.retraining_count_,
        "buffered_samples": model.buffered_sample_count_,
    }


def main() -> None:
    data_path = "demo/data/titanic.csv"
    header, rows = load_csv(data_path)
    numeric_columns = ("pclass", "age", "sibsp", "parch", "fare")
    X_raw = rows[:, [header.index(name) for name in numeric_columns]].astype(float)
    X = Imputer(strategy="mean").fit_transform(X_raw)
    y = rows[:, header.index("survived")].astype(int)
    threshold = float(np.mean(X.sum(axis=1)))
    timing = compare_implementations(
        {
            "Python loops": loop_threshold_predict,
            "NumPy vectorized": vectorized_threshold_predict,
        },
        X,
        threshold,
        reference="Python loops",
        repeats=3,
        warmup=1,
    )
    stream_chunk_size = 32
    buffer_rows = []
    for size in (32, 64, 128, 256):
        timing_for_size = benchmark(
            evaluate_buffer_size,
            X,
            y,
            size,
            stream_chunk_size,
            repeats=2,
            warmup=0,
        )
        measurement = evaluate_buffer_size(X, y, size, stream_chunk_size)
        buffer_rows.append(
            {
                **measurement,
                "median_training_seconds": timing_for_size["median_seconds"],
            }
        )
    print(json.dumps(
        {
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "dataset": "demo/data/titanic.csv",
            "samples": int(X.shape[0]),
            "numeric_features": list(numeric_columns),
            "missing_age_imputation": "whole-file mean (benchmark setup only)",
            "loop_vs_vectorized": timing,
            "buffer_size_experiments": buffer_rows,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
