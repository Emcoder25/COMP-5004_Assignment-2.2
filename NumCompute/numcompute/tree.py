from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np


class DecisionTreeClassifier:
    """A lightweight, depth-limited decision tree classifier.

    The tree supports buffered retraining for streaming workloads. Each call to
    ``partial_fit`` appends a chunk into an internal buffer and retrains the tree
    once the buffer reaches ``buffer_size`` samples.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        buffer_size: int = 100,
        criterion: str = "gini",
    ) -> None:
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = int(buffer_size)
        self.criterion = criterion
        self._buffer_X: list[np.ndarray] = []
        self._buffer_y: list[np.ndarray] = []
        self.root_: dict[str, Any] | None = None
        self.n_features_in_: int | None = None
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        counts = np.bincount(y.astype(int, copy=False))
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs**2)

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        counts = np.bincount(y.astype(int, copy=False))
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == "gini":
            return self._gini(y)
        return self._entropy(y)

    def _feature_indices(self, n_features: int) -> np.ndarray:
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        rng = np.random.default_rng(0)
        return rng.choice(np.arange(n_features), size=self.max_features, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, np.ndarray | None, np.ndarray | None, float]:
        if X.shape[0] == 0:
            return None, None, None, None, 0.0

        current_score = self._impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        feature_indices = self._feature_indices(X.shape[1])
        for feature_idx in feature_indices:
            values = X[:, feature_idx]
            unique_values = np.unique(values)
            if unique_values.size == 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]
                weighted_impurity = (
                    (left_y.size / y.size) * self._impurity(left_y)
                    + (right_y.size / y.size) * self._impurity(right_y)
                )
                gain = current_score - weighted_impurity
                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(feature_idx)
                    best_threshold = float(threshold)
                    best_left = left_mask
                    best_right = right_mask

        return best_feature, best_threshold, best_left, best_right, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict[str, Any]:
        if y.size == 0:
            return {"leaf": True, "prediction": 0, "count": 0}

        if depth >= (self.max_depth if self.max_depth is not None else np.inf):
            label = Counter(y.tolist()).most_common(1)[0][0]
            return {"leaf": True, "prediction": label, "count": int(y.size)}

        if y.size < self.min_samples_split:
            label = Counter(y.tolist()).most_common(1)[0][0]
            return {"leaf": True, "prediction": label, "count": int(y.size)}

        if np.unique(y).size == 1:
            return {"leaf": True, "prediction": int(y[0]), "count": int(y.size)}

        feature, threshold, left_mask, right_mask, gain = self._best_split(X, y)
        if feature is None or threshold is None or left_mask is None or right_mask is None or gain <= 0.0:
            label = Counter(y.tolist()).most_common(1)[0][0]
            return {"leaf": True, "prediction": label, "count": int(y.size)}

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf": False,
            "feature": int(feature),
            "threshold": float(threshold),
            "left": left_tree,
            "right": right_tree,
            "count": int(y.size),
        }

    def _validate_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if y.size == 0:
            raise ValueError("y must not be empty")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X_arr, y_arr = self._validate_X_y(X, y)
        self.root_ = self._build_tree(X_arr, y_arr, depth=0)
        return self

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> "DecisionTreeClassifier":
        X_chunk_arr = np.asarray(X_chunk, dtype=float)
        y_chunk_arr = np.asarray(y_chunk)
        if X_chunk_arr.ndim != 2:
            raise ValueError("X_chunk must be a 2D array")
        if y_chunk_arr.ndim != 1:
            raise ValueError("y_chunk must be a 1D array")
        if X_chunk_arr.shape[0] != y_chunk_arr.shape[0]:
            raise ValueError("X_chunk and y_chunk must contain the same number of samples")

        self._buffer_X.append(X_chunk_arr)
        self._buffer_y.append(y_chunk_arr)

        total_X = np.concatenate(self._buffer_X, axis=0) if self._buffer_X else np.empty((0, 0))
        total_y = np.concatenate(self._buffer_y, axis=0) if self._buffer_y else np.empty((0,))

        if total_X.size > 0 and total_X.shape[0] >= self.buffer_size:
            self.fit(total_X, total_y)
            self._buffer_X.clear()
            self._buffer_y.clear()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.root_ is None:
            raise ValueError("The decision tree has not been fitted yet.")
        return np.array([self._predict_one(sample, self.root_) for sample in X_arr], dtype=int)

    def _predict_one(self, sample: np.ndarray, node: dict[str, Any]) -> int:
        if node["leaf"]:
            return int(node["prediction"])
        if sample[node["feature"]] <= node["threshold"]:
            return self._predict_one(sample, node["left"])
        return self._predict_one(sample, node["right"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if self.root_ is None:
            raise ValueError("The decision tree has not been fitted yet.")
        probs = []
        for sample in X_arr:
            probs.append(self._predict_proba_one(sample, self.root_))
        return np.asarray(probs)

    def _predict_proba_one(self, sample: np.ndarray, node: dict[str, Any]) -> np.ndarray:
        if node["leaf"]:
            labels = self.classes_ if self.classes_ is not None else np.array([node["prediction"]])
            probs = np.zeros(labels.shape[0], dtype=float)
            probs[np.where(labels == node["prediction"])[0][0]] = 1.0
            return probs
        if sample[node["feature"]] <= node["threshold"]:
            return self._predict_proba_one(sample, node["left"])
        return self._predict_proba_one(sample, node["right"])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.root_ is None:
            raise ValueError("The decision tree has not been fitted yet.")
        predictions = self.predict(X)
        return float(np.mean(predictions == np.asarray(y)))


__all__ = ["DecisionTreeClassifier"]
