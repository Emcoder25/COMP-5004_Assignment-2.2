"""Batch and incremental classification metrics.

Streaming AUC uses bounded score bins and is approximate: score resolution is
limited to ``n_bins`` and each bin treats positive/negative ties as half credit.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _labels(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError(f"labels must be 1D; received shapes {true.shape} and {pred.shape}.")
    if true.size == 0:
        raise ValueError("labels must not be empty.")
    if true.shape != pred.shape:
        raise ValueError(f"label shapes must match; received {true.shape} and {pred.shape}.")
    if true.dtype.kind in "fc" and not np.all(np.isfinite(true)):
        raise ValueError("y_true must not contain NaN or infinity.")
    if pred.dtype.kind in "fc" and not np.all(np.isfinite(pred)):
        raise ValueError("y_pred must not contain NaN or infinity.")
    if true.dtype.kind == "O" and any(value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) for value in true):
        raise ValueError("y_true must not contain missing labels.")
    if pred.dtype.kind == "O" and any(value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) for value in pred):
        raise ValueError("y_pred must not contain missing labels.")
    return true, pred


def confusion_matrix(y_true: Any, y_pred: Any, labels: Any = None) -> np.ndarray:
    true, pred = _labels(y_true, y_pred)
    classes = np.unique(np.concatenate((true, pred))) if labels is None else np.asarray(labels)
    if classes.ndim != 1 or classes.size == 0 or np.unique(classes).size != classes.size:
        raise ValueError("labels must be a non-empty 1D sequence of unique values.")
    true_matches = true[:, None] == classes[None, :]
    pred_matches = pred[:, None] == classes[None, :]
    if not np.all(np.any(true_matches, axis=1)) or not np.all(np.any(pred_matches, axis=1)):
        raise ValueError("labels must contain every observed actual and predicted class.")
    ti, pi = np.argmax(true_matches, axis=1), np.argmax(pred_matches, axis=1)
    return np.bincount(ti * classes.size + pi, minlength=classes.size**2).reshape(classes.size, classes.size)


def accuracy(y_true: Any, y_pred: Any) -> float:
    true, pred = _labels(y_true, y_pred)
    return float(np.mean(true == pred))


def _class_score(y_true: Any, y_pred: Any, metric: str, *, average: str = "binary", positive_label: Any = 1, zero_division: float = 0.0) -> float:
    true, pred = _labels(y_true, y_pred)
    cm = confusion_matrix(true, pred)
    classes = np.unique(np.concatenate((true, pred)))
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp if metric == "precision" else tp + fn
    per_class = np.divide(tp, denom, out=np.full(tp.shape, float(zero_division)), where=denom != 0)
    if metric == "f1":
        pden = tp + fp
        rden = tp + fn
        p = np.divide(tp, pden, out=np.full(tp.shape, float(zero_division)), where=pden != 0)
        r = np.divide(tp, rden, out=np.full(tp.shape, float(zero_division)), where=rden != 0)
        per_class = np.divide(2 * p * r, p + r, out=np.full(tp.shape, float(zero_division)), where=(p + r) != 0)
    if average == "binary":
        idx = np.flatnonzero(classes == positive_label)
        if idx.size != 1:
            raise ValueError(f"positive_label {positive_label!r} must be present in the labels.")
        return float(per_class[idx[0]])
    if average == "micro":
        total_tp = float(tp.sum())
        total = float(tp.sum() + (fp.sum() if metric == "precision" else fn.sum()))
        if metric == "f1":
            p = total_tp / (total_tp + fp.sum()) if total_tp + fp.sum() else zero_division
            r = total_tp / (total_tp + fn.sum()) if total_tp + fn.sum() else zero_division
            return float(2 * p * r / (p + r)) if p + r else float(zero_division)
        return total_tp / total if total else float(zero_division)
    if average == "macro":
        return float(np.mean(per_class))
    if average == "weighted":
        weights = cm.sum(axis=1)
        return float(np.average(per_class, weights=weights))
    raise ValueError("average must be 'binary', 'micro', 'macro', or 'weighted'.")


def precision(y_true: Any, y_pred: Any, positive_label: Any = 1, *, average: str = "binary", zero_division: float = 0.0) -> float:
    return _class_score(y_true, y_pred, "precision", average=average, positive_label=positive_label, zero_division=zero_division)


def recall(y_true: Any, y_pred: Any, positive_label: Any = 1, *, average: str = "binary", zero_division: float = 0.0) -> float:
    return _class_score(y_true, y_pred, "recall", average=average, positive_label=positive_label, zero_division=zero_division)


def f1(y_true: Any, y_pred: Any, positive_label: Any = 1, *, average: str = "binary", zero_division: float = 0.0) -> float:
    return _class_score(y_true, y_pred, "f1", average=average, positive_label=positive_label, zero_division=zero_division)


def roc_curve(y_true: Any, y_score: Any, positive_label: Any = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, scores = np.asarray(y_true), np.asarray(y_score)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or labels.size == 0:
        raise ValueError(f"y_true and y_score must be non-empty 1D arrays of equal shape; received {labels.shape} and {scores.shape}.")
    if scores.dtype.kind not in "iuf":
        raise TypeError(f"y_score must be real numeric; received dtype {scores.dtype}.")
    scores = scores.astype(float, copy=False)
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must contain only finite values.")
    classes = np.unique(labels)
    if classes.size != 2 or not np.any(labels == positive_label):
        raise ValueError("ROC requires exactly two classes and positive_label present in y_true.")
    positive = labels == positive_label
    order = np.argsort(-scores, kind="stable")
    s, p = scores[order], positive[order]
    distinct = np.r_[np.flatnonzero(np.diff(s) != 0), s.size - 1]
    tps, fps = np.cumsum(p)[distinct], np.cumsum(~p)[distinct]
    return np.r_[0.0, fps / np.count_nonzero(~positive)], np.r_[0.0, tps / np.count_nonzero(positive)], np.r_[np.inf, s[distinct]]


def auc(x: Any, y: Any) -> float:
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size or x_arr.size < 2:
        raise ValueError("x and y must be equal-length 1D arrays with at least two points.")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("x and y must contain only finite values.")
    differences = np.diff(x_arr)
    if not (np.all(differences >= 0) or np.all(differences <= 0)):
        raise ValueError("x must be monotonic.")
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(abs(trapezoid(y_arr, x_arr)))


def roc_auc_score(y_true: Any, y_score: Any, positive_label: Any = 1) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score, positive_label)
    return auc(fpr, tpr)


class StreamingMetric:
    """Rolling-or-cumulative confusion-count metric base."""

    def __init__(self, *, window: int | None = None, labels: Any = None, positive_label: Any = 1, average: str = "binary") -> None:
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 1):
            raise ValueError("window must be a positive integer or None.")
        self.window, self.labels = window, None if labels is None else np.asarray(labels)
        self.positive_label, self.average = positive_label, average
        self._chunks: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self.reset()

    def reset(self) -> None:
        self._chunks.clear()
        self._true: list[np.ndarray] = []
        self._pred: list[np.ndarray] = []

    def update(self, y_true: Any, y_pred: Any) -> None:
        true, pred = _labels(y_true, y_pred)
        self._chunks.append((true.copy(), pred.copy()))
        if self.window is not None:
            retained = sum(chunk[0].size for chunk in self._chunks)
            while retained > self.window:
                first_true, first_pred = self._chunks.popleft()
                excess = retained - self.window
                if excess < first_true.size:
                    self._chunks.appendleft((first_true[excess:], first_pred[excess:]))
                    retained -= excess
                else:
                    retained -= first_true.size
        if self.window is None:
            self._true.append(true.copy())
            self._pred.append(pred.copy())

    def _values(self) -> tuple[np.ndarray, np.ndarray]:
        chunks = list(self._chunks) if self.window is not None else list(zip(self._true, self._pred))
        if not chunks:
            return np.array([]), np.array([])
        return np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])


class Accuracy(StreamingMetric):
    def result(self) -> float:
        true, pred = self._values()
        return accuracy(true, pred) if true.size else 0.0


class Precision(StreamingMetric):
    def result(self) -> float:
        true, pred = self._values()
        return precision(true, pred, self.positive_label, average=self.average) if true.size else 0.0


class Recall(StreamingMetric):
    def result(self) -> float:
        true, pred = self._values()
        return recall(true, pred, self.positive_label, average=self.average) if true.size else 0.0


class F1Score(StreamingMetric):
    def result(self) -> float:
        true, pred = self._values()
        return f1(true, pred, self.positive_label, average=self.average) if true.size else 0.0


class ConfusionMatrix(StreamingMetric):
    def result(self) -> np.ndarray:
        true, pred = self._values()
        if true.size == 0:
            return np.zeros((0, 0), dtype=np.int64)
        return confusion_matrix(true, pred, self.labels)


class AUC:
    """Bounded-memory binned streaming ROC AUC, approximate at bin resolution."""

    def __init__(self, *, n_bins: int = 1024, positive_label: Any = 1) -> None:
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2.")
        self.n_bins, self.positive_label = n_bins, positive_label
        self.reset()

    def reset(self) -> None:
        self.positive_counts = np.zeros(self.n_bins, dtype=np.int64)
        self.negative_counts = np.zeros(self.n_bins, dtype=np.int64)

    def update(self, y_true: Any, y_score: Any) -> None:
        labels, scores = np.asarray(y_true), np.asarray(y_score)
        if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or labels.size == 0:
            raise ValueError("y_true and y_score must be non-empty 1D arrays of equal shape.")
        if scores.dtype.kind not in "iuf":
            raise TypeError(f"y_score must be real numeric; received dtype {scores.dtype}.")
        scores = scores.astype(float, copy=False)
        if not np.all(np.isfinite(scores)):
            raise ValueError("y_score must contain only finite values.")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("streaming AUC scores must be probabilities in [0, 1].")
        bins = np.minimum((scores * self.n_bins).astype(int), self.n_bins - 1)
        positive = labels == self.positive_label
        self.positive_counts += np.bincount(bins[positive], minlength=self.n_bins)
        self.negative_counts += np.bincount(bins[~positive], minlength=self.n_bins)

    def result(self) -> float:
        positives, negatives = int(self.positive_counts.sum()), int(self.negative_counts.sum())
        if not positives or not negatives:
            raise ValueError("AUC requires at least one positive and one negative observation.")
        lower_negative = np.cumsum(self.negative_counts) - self.negative_counts
        concordant = np.sum(self.positive_counts * (lower_negative + 0.5 * self.negative_counts))
        return float(concordant / (positives * negatives))


__all__ = [
    "accuracy", "precision", "recall", "f1", "confusion_matrix",
    "roc_curve", "auc", "roc_auc_score", "StreamingMetric", "Accuracy",
    "Precision", "Recall", "F1Score", "ConfusionMatrix", "AUC",
    "StreamingAccuracy", "StreamingPrecision", "StreamingRecall",
    "StreamingF1", "StreamingConfusionMatrix", "StreamingAUC",
]

StreamingAccuracy = Accuracy
StreamingPrecision = Precision
StreamingRecall = Recall
StreamingF1 = F1Score
StreamingConfusionMatrix = ConfusionMatrix
StreamingAUC = AUC
