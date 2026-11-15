"""NumPy decision-tree classifier with buffered incremental retraining."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ._validation import positive_int, supervised_chunk


class DecisionTreeClassifier:
    """Depth-limited classification tree using Gini or entropy splits.

    ``partial_fit`` buffers chunks and retrains after reaching ``buffer_size``.
    A chunk that crosses the threshold is used in full. Until a retrain, a
    fitted estimator keeps predicting with the previous tree.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | str | None = None,
        buffer_size: int = 100,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        if max_depth is not None:
            if isinstance(max_depth, (bool, np.bool_)) or not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
                raise ValueError("max_depth must be a non-negative integer or None.")
            max_depth = int(max_depth)
        self.min_samples_split = positive_int(min_samples_split, "min_samples_split")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        self.buffer_size = positive_int(buffer_size, "buffer_size")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_features not in (None, "sqrt", "log2") and (
            isinstance(max_features, (bool, np.bool_))
            or not isinstance(max_features, (int, np.integer))
            or max_features < 1
        ):
            raise ValueError("max_features must be a positive integer, 'sqrt', 'log2', or None.")
        self.max_depth, self.max_features, self.criterion = max_depth, max_features, criterion
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.nodes_: list[dict[str, Any]] = []
        self.n_features_in_: int | None = None
        self._buffer_X: list[np.ndarray] = []
        self._buffer_y: list[np.ndarray] = []
        self.retraining_count_ = 0

    @property
    def is_fitted_(self) -> bool:
        return bool(self.nodes_)

    @property
    def buffered_sample_count_(self) -> int:
        return sum(chunk.shape[0] for chunk in self._buffer_X)

    @property
    def depth_(self) -> int:
        return max((node["depth"] for node in self.nodes_), default=0)

    @property
    def n_leaves_(self) -> int:
        return sum(bool(node["leaf"]) for node in self.nodes_)

    def _impurity(self, counts: np.ndarray) -> np.ndarray:
        totals = counts.sum(axis=-1, keepdims=True)
        probabilities = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities**2, axis=-1)
        terms = np.zeros_like(probabilities)
        positive = probabilities > 0
        terms[positive] = probabilities[positive] * np.log2(probabilities[positive])
        return -terms.sum(axis=-1)

    def _feature_subset(self, width: int, rng: np.random.Generator) -> np.ndarray:
        if self.max_features is None:
            count = width
        elif self.max_features == "sqrt":
            count = max(1, int(np.sqrt(width)))
        elif self.max_features == "log2":
            count = max(1, int(np.log2(width)))
        else:
            count = min(int(self.max_features), width)
        if count == width:
            return np.arange(width)
        return np.sort(rng.choice(width, size=count, replace=False))

    def _best_split(self, X: np.ndarray, encoded_y: np.ndarray, rows: np.ndarray, rng: np.random.Generator):
        labels_count = len(self.classes_)
        parent_counts = np.bincount(encoded_y[rows], minlength=labels_count)
        parent_impurity = float(self._impurity(parent_counts))
        best_gain, best = 0.0, None
        for feature in self._feature_subset(X.shape[1], rng):
            order = rows[np.argsort(X[rows, feature], kind="stable")]
            sorted_values = X[order, feature]
            sorted_labels = encoded_y[order]
            boundaries = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
            if boundaries.size == 0:
                continue
            one_hot = np.eye(labels_count, dtype=np.int64)[sorted_labels]
            cumulative = np.cumsum(one_hot, axis=0)
            left_counts = cumulative[boundaries]
            right_counts = parent_counts - left_counts
            n_left = boundaries + 1
            n_total = rows.size
            weighted = (n_left / n_total) * self._impurity(left_counts) + ((n_total - n_left) / n_total) * self._impurity(right_counts)
            gains = parent_impurity - weighted
            local = int(np.argmax(gains))
            if gains[local] > best_gain:
                boundary = int(boundaries[local])
                lo, hi = sorted_values[boundary], sorted_values[boundary + 1]
                threshold = float(lo / 2.0 + hi / 2.0)
                best_gain = float(gains[local])
                best = (int(feature), threshold)
        return best

    def fit(self, X: Any, y: Any) -> "DecisionTreeClassifier":
        features, labels = supervised_chunk(X, y)
        if features.ndim == 1:
            features = features[:, None]
        if features.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if features.dtype.kind not in "iuf":
            raise TypeError(f"X must be numeric; received dtype {features.dtype}.")
        features = features.astype(float, copy=False)
        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values.")
        if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
            raise ValueError("y must contain only finite labels.")
        classes, encoded_y = np.unique(labels, return_inverse=True)
        self.classes_ = classes
        self.n_features_in_ = features.shape[1]
        self.nodes_ = [{"leaf": True, "depth": 0, "prediction": int(np.bincount(encoded_y).argmax()), "counts": np.bincount(encoded_y, minlength=classes.size)}]
        rng = np.random.default_rng(self.random_state)
        pending: list[tuple[int, np.ndarray, int]] = [(0, np.arange(len(labels)), 0)]
        while pending:
            node_index, rows, depth = pending.pop()
            counts = np.bincount(encoded_y[rows], minlength=classes.size)
            prediction = int(np.argmax(counts))
            node = self.nodes_[node_index]
            node.update(prediction=prediction, counts=counts, depth=depth)
            if (
                np.unique(encoded_y[rows]).size <= 1
                or rows.size < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
            ):
                continue
            split = self._best_split(features, encoded_y, rows, rng)
            if split is None:
                continue
            feature, threshold = split
            left_rows = rows[features[rows, feature] <= threshold]
            right_rows = rows[features[rows, feature] > threshold]
            if left_rows.size == 0 or right_rows.size == 0:
                continue
            left_index, right_index = len(self.nodes_), len(self.nodes_) + 1
            node.update(leaf=False, feature=feature, threshold=threshold, left=left_index, right=right_index)
            self.nodes_.extend(
                [
                    {"leaf": True, "depth": depth + 1, "prediction": prediction, "counts": counts},
                    {"leaf": True, "depth": depth + 1, "prediction": prediction, "counts": counts},
                ]
            )
            pending.append((right_index, right_rows, depth + 1))
            pending.append((left_index, left_rows, depth + 1))
        self._buffer_X.clear()
        self._buffer_y.clear()
        self.retraining_count_ += 1
        return self

    def partial_fit(self, X: Any, y: Any) -> "DecisionTreeClassifier":
        features, labels = supervised_chunk(X, y)
        if features.ndim == 1:
            features = features[:, None]
        if features.dtype.kind not in "iuf":
            raise TypeError(f"X must be numeric; received dtype {features.dtype}.")
        features = features.astype(float, copy=False)
        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values.")
        if self._buffer_X:
            width = max(features.shape[1], self._buffer_X[0].shape[1])
            if features.shape[1] != width:
                features = np.pad(features, ((0, 0), (0, width - features.shape[1])))
                self._buffer_X = [np.pad(chunk, ((0, 0), (0, width - chunk.shape[1]))) for chunk in self._buffer_X]
        self._buffer_X.append(features.copy())
        self._buffer_y.append(labels.copy())
        if self.buffered_sample_count_ >= self.buffer_size:
            self.fit(np.concatenate(self._buffer_X), np.concatenate(self._buffer_y))
        return self

    def _validate_predict_X(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("DecisionTreeClassifier is not fitted; call fit or buffer enough samples with partial_fit.")
        features = np.asarray(X)
        if features.ndim == 1:
            features = features[:, None]
        if features.ndim != 2 or features.shape[1] < self.n_features_in_:
            raise ValueError(f"X must have at least {self.n_features_in_} features; received shape {features.shape}, dtype {features.dtype}.")
        if features.dtype.kind not in "iuf":
            raise TypeError(f"X must be numeric; received dtype {features.dtype}.")
        features = features.astype(float, copy=False)
        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values.")
        return features

    def _leaf_ids(self, X: np.ndarray) -> np.ndarray:
        node_ids = np.zeros(X.shape[0], dtype=int)
        active = np.ones(X.shape[0], dtype=bool)
        while np.any(active):
            for node_id in np.unique(node_ids[active]):
                rows = np.flatnonzero(active & (node_ids == node_id))
                node = self.nodes_[int(node_id)]
                if node["leaf"]:
                    active[rows] = False
                    continue
                go_left = X[rows, node["feature"]] <= node["threshold"]
                node_ids[rows[go_left]] = node["left"]
                node_ids[rows[~go_left]] = node["right"]
        return node_ids

    def predict(self, X: Any) -> np.ndarray:
        features = self._validate_predict_X(X)
        leaves = self._leaf_ids(features)
        return self.classes_[np.asarray([self.nodes_[i]["prediction"] for i in leaves], dtype=int)]

    def predict_proba(self, X: Any) -> np.ndarray:
        features = self._validate_predict_X(X)
        if features.shape[0] == 0:
            return np.empty((0, self.classes_.size), dtype=float)
        leaves = self._leaf_ids(features)
        counts = np.vstack([self.nodes_[i]["counts"] for i in leaves]).astype(float)
        return np.divide(counts, counts.sum(axis=1, keepdims=True), out=np.zeros_like(counts), where=counts.sum(axis=1, keepdims=True) != 0)

    def score(self, X: Any, y: Any) -> float:
        predictions = self.predict(X)
        targets = np.asarray(y)
        if targets.ndim != 1 or targets.shape != predictions.shape:
            raise ValueError(f"y must have shape {predictions.shape}; received shape {targets.shape}.")
        return float(np.mean(predictions == targets))


DecisionTree = DecisionTreeClassifier

__all__ = ["DecisionTreeClassifier", "DecisionTree"]
