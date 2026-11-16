"""Train and evaluate a NumCompute-Stream classifier on the local Titanic CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from numcompute_stream.io import load_csv
from numcompute_stream.metrics import accuracy, confusion_matrix
from numcompute_stream.pipeline import FeatureUnion, Pipeline
from numcompute_stream.preprocessing import (
    ColumnSelector,
    Imputer,
    OneHotEncoder,
    StandardScaler,
)
from numcompute_stream.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).parent / "data" / "titanic.csv"
FEATURE_COLUMNS = ("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked")


def load_titanic_dataset(path: str | Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load predictors and the ``survived`` target, excluding target duplicates."""
    header, rows = load_csv(path)
    required = (*FEATURE_COLUMNS, "survived")
    missing_columns = [name for name in required if name not in header]
    if missing_columns:
        raise ValueError(f"Titanic CSV is missing required columns: {missing_columns}.")
    feature_indices = [header.index(name) for name in FEATURE_COLUMNS]
    X = rows[:, feature_indices]
    y = rows[:, header.index("survived")].astype(int)
    return X, y


def make_titanic_pipeline() -> Pipeline:
    """Build the mixed numeric/categorical preprocessing and tree pipeline."""
    features = FeatureUnion(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("select", ColumnSelector([0, 2, 3, 4, 5])),
                        ("impute", Imputer(strategy="mean")),
                        ("scale", StandardScaler()),
                    ]
                ),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("select", ColumnSelector([1, 6])),
                        ("impute", Imputer(strategy="constant", fill_value="Unknown")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("tree", DecisionTreeClassifier(max_depth=5, random_state=7)),
        ]
    )


def train_titanic(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_path: str | Path = DATA_PATH,
) -> dict[str, Any]:
    """Fit on a deterministic train split and report metrics on held-out passengers."""
    if not np.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("test_size must be finite and strictly between 0 and 1.")
    X, y = load_titanic_dataset(dataset_path)
    if y.size < 2:
        raise ValueError("Titanic CSV must contain at least two passenger rows.")

    permutation = np.random.default_rng(random_state).permutation(y.size)
    n_test = min(y.size - 1, max(1, int(np.ceil(test_size * y.size))))
    test_indices, train_indices = permutation[:n_test], permutation[n_test:]

    model = make_titanic_pipeline()
    model.fit(X[train_indices], y[train_indices])
    predictions = model.predict(X[test_indices])
    return {
        "model": model,
        "accuracy": accuracy(y[test_indices], predictions),
        "confusion_matrix": confusion_matrix(y[test_indices], predictions, labels=[0, 1]),
        "train_samples": int(train_indices.size),
        "test_samples": int(test_indices.size),
        "feature_columns": FEATURE_COLUMNS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    result = train_titanic(
        test_size=args.test_size,
        random_state=args.random_state,
        dataset_path=args.data,
    )
    print(f"Dataset: {args.data}")
    print(f"Features: {', '.join(result['feature_columns'])}")
    print(f"Train/test passengers: {result['train_samples']}/{result['test_samples']}")
    print(f"Held-out accuracy: {result['accuracy']:.3f}")
    print("Confusion matrix (rows=true, columns=predicted; labels=[0, 1]):")
    print(result["confusion_matrix"])


if __name__ == "__main__":
    main()
