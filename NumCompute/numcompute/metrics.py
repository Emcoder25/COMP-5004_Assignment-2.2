import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

_ZERO_DIVISION_UNSET = object()


def _validate_labels(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> tuple[NDArray, NDArray]:
    """
    Validate true and predicted labels.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    tuple of numpy.ndarray
        Validated true and predicted label arrays.

    Raises
    ------
    ValueError
        If either input is not one-dimensional, is empty, has a
        mismatched shape, or contains NaN or infinity in a
        floating-point or complex array.

    Complexity: O(n) time and O(n) space for n labels.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shapes")

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")

    if true.size == 0 or pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if true.dtype.kind in "fc" and not np.all(np.isfinite(true)):
        raise ValueError("y_true must not contain NaN or infinity")

    if pred.dtype.kind in "fc" and not np.all(np.isfinite(pred)):
        raise ValueError("y_pred must not contain NaN or infinity")

    return true, pred


def _safe_divide(
    numerator: int | float | np.integer | np.floating,
    denominator: int | float | np.integer | np.floating,
    zero_division: int = 0,
) -> float:
    """
    Divide two values while handling a zero denominator.

    Parameters
    ----------
    numerator : int or float
        Numerator of the division.
    denominator : int or float
        Denominator of the division.
    zero_division : int, default=0
        Value to return when the denominator is zero. Must be 0 or 1.

    Returns
    -------
    float
        Division result, or zero_division if the denominator is zero.

    Raises
    ------
    ValueError
        If zero_division is not 0 or 1.

    Complexity: O(1) time and O(1) space.
    """
    if zero_division is _ZERO_DIVISION_UNSET:
        zero_division = 0
    if not isinstance(zero_division, (int, float, np.number)):
        raise ValueError("zero_division must be numeric")

    if denominator == 0:
        return float(zero_division)

    return float(numerator / denominator)


def accuracy(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> float:
    """
    Calculate the fraction of correct predictions.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels, with the same shape as y_true.

    Returns
    -------
    float
        Number of correct predictions divided by total predictions.

    Raises
    ------
    ValueError
        If the arrays are empty, not one-dimensional, or have
        different shapes.

    Complexity: O(n) time for n predictions.
    """
    true, pred = _validate_labels(y_true, y_pred)

    return float(np.mean(true == pred))


def confusion_matrix(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    labels: npt.ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Calculate the confusion matrix for classification predictions.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.
    labels : array-like, optional
        Ordered list of class labels to include. If omitted, observed
        labels from both arrays are sorted automatically.

    Returns
    -------
    numpy.ndarray
        Integer matrix where rows represent actual classes and columns
        represent predicted classes.

    Raises
    ------
    ValueError
        If the input arrays are invalid, labels are empty or duplicated,
        or an observed label is missing from the provided labels.

    Complexity: O(n + k^2) time, where n is the number of samples and
    k is the number of classes.
    """
    true, pred = _validate_labels(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((true, pred)))
    else:
        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")

        if labels.size == 0:
            raise ValueError("labels must not be empty")

        if np.unique(labels).size != labels.size:
            raise ValueError("labels must not contain duplicates")

    observed = np.unique(np.concatenate((true, pred)))
    if labels.size and any(label not in labels for label in observed):
        raise ValueError("labels must include every observed class.")

    label_to_index = {
        label: i for i, label in enumerate(labels.tolist())
    }
    matrix = np.zeros((labels.size, labels.size), dtype=np.int64)

    for actual, predicted in zip(true, pred):
        if actual not in label_to_index:
            raise ValueError(f"actual label {actual!r} is not in labels")

        if predicted not in label_to_index:
            raise ValueError(
                f"predicted label {predicted!r} is not in labels"
            )

        matrix[
            label_to_index[actual],
            label_to_index[predicted],
        ] += 1

    return matrix


def precision(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    positive_label: object = 1,
    average: str | None = None,
    zero_division: object = _ZERO_DIVISION_UNSET,
) -> float:
    true, pred = _validate_labels(y_true, y_pred)
    if average is not None:
        if average not in {"micro", "macro", "weighted"}:
            raise ValueError("average must be micro, macro, or weighted.")
        labels = np.unique(np.concatenate((true, pred)))
        if average == "micro":
            return float(np.mean(true == pred))
        fallback = 0 if zero_division is _ZERO_DIVISION_UNSET else zero_division
        values = [precision(true, pred, positive_label=label, zero_division=fallback) for label in labels]
        if average == "weighted":
            weights = [np.count_nonzero(true == label) for label in labels]
            return float(np.average(values, weights=weights))
        return float(np.mean(values))
    if not np.any(true == positive_label):
        if zero_division is _ZERO_DIVISION_UNSET:
            raise ValueError("positive_label must be present in y_true.")
        return float(zero_division)
    actual_positive = true == positive_label
    predicted_positive = pred == positive_label
    tp = np.count_nonzero(actual_positive & predicted_positive)
    fp = np.count_nonzero(~actual_positive & predicted_positive)
    return _safe_divide(tp, tp + fp, zero_division)


def recall(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    positive_label: object = 1,
    average: str | None = None,
    zero_division: object = _ZERO_DIVISION_UNSET,
) -> float:
    true, pred = _validate_labels(y_true, y_pred)
    if average is not None:
        if average not in {"micro", "macro", "weighted"}:
            raise ValueError("average must be micro, macro, or weighted.")
        labels = np.unique(np.concatenate((true, pred)))
        if average == "micro":
            return float(np.mean(true == pred))
        fallback = 0 if zero_division is _ZERO_DIVISION_UNSET else zero_division
        values = [recall(true, pred, positive_label=label, zero_division=fallback) for label in labels]
        if average == "weighted":
            weights = [np.count_nonzero(true == label) for label in labels]
            return float(np.average(values, weights=weights))
        return float(np.mean(values))
    if not np.any(true == positive_label):
        if zero_division is _ZERO_DIVISION_UNSET:
            raise ValueError("positive_label must be present in y_true.")
        return float(zero_division)
    actual_positive = true == positive_label
    predicted_positive = pred == positive_label

    tp = np.count_nonzero(actual_positive & predicted_positive)
    fn = np.count_nonzero(actual_positive & ~predicted_positive)

    return _safe_divide(tp, tp + fn, zero_division)


def f1(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    positive_label: object = 1,
    average: str | None = None,
    zero_division: object = _ZERO_DIVISION_UNSET,
) -> float:
    true, pred = _validate_labels(y_true, y_pred)
    if average is not None:
        if average not in {"micro", "macro", "weighted"}:
            raise ValueError("average must be micro, macro, or weighted.")
        labels = np.unique(np.concatenate((true, pred)))
        if average == "micro":
            return float(np.mean(true == pred))
        fallback = 0 if zero_division is _ZERO_DIVISION_UNSET else zero_division
        values = [f1(true, pred, positive_label=label, zero_division=fallback) for label in labels]
        if average == "weighted":
            weights = [np.count_nonzero(true == label) for label in labels]
            return float(np.average(values, weights=weights))
        return float(np.mean(values))
    if not np.any(true == positive_label):
        if zero_division is _ZERO_DIVISION_UNSET:
            raise ValueError("positive_label must be present in y_true.")
        return float(zero_division)
    p = precision(
        y_true,
        y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )
    r = recall(
        y_true,
        y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )

    return _safe_divide(2 * p * r, p + r, zero_division)


def mse(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> float:
    """
    Calculate mean squared error for regression predictions.

    Parameters
    ----------
    y_true : array-like
        Actual numeric target values.
    y_pred : array-like
        Predicted numeric target values, matching y_true in shape.

    Returns
    -------
    float
        Mean of the squared differences between actual and predicted
        values.

    Raises
    ------
    TypeError
        If either input contains non-real numeric values.
    ValueError
        If the arrays have different shapes, are empty, or contain
        NaN or infinity.

    Complexity: O(n) time for n target values.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.dtype.kind not in "iuf" or pred.dtype.kind not in "iuf":
        raise TypeError("y_true and y_pred must contain real numeric values")

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true and y_pred must not contain NaN or infinity")

    true = true.astype(float, copy=False)
    pred = pred.astype(float, copy=False)

    return float(np.mean((true - pred) ** 2))


def roc_curve(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    positive_label: object = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate ROC curve points for binary classification.

    Parameters
    ----------
    y_true : array-like
        Actual binary class labels.
    y_score : array-like
        Numeric scores for the positive class. Higher values indicate
        a stronger prediction of the positive class.
    positive_label : object, default=1
        Label treated as the positive class.

    Returns
    -------
    tuple of numpy.ndarray
        False positive rates, true positive rates, and descending
        classification thresholds. The first threshold is infinity.

    Raises
    ------
    TypeError
        If y_score contains non-real numeric values.
    ValueError
        If inputs are empty, not one-dimensional, have different shapes,
        contain non-finite values, or y_true does not contain exactly
        two classes.

    Complexity: O(n log n) time due to sorting n prediction scores.
    """
    true = np.asarray(y_true)
    scores = np.asarray(y_score)

    if true.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays")

    if true.size == 0 or scores.size == 0:
        raise ValueError("y_true and y_score must not be empty")

    if true.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")

    if scores.dtype.kind not in "iuf":
        raise TypeError("y_score must contain real numeric values")

    scores = scores.astype(float, copy=False)

    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must not contain NaN or infinity; finite values are required")

    if true.dtype.kind in "fc" and not np.all(np.isfinite(true)):
        raise ValueError("y_true must not contain NaN or infinity")

    classes = np.unique(true)

    if classes.size != 2:
        raise ValueError("ROC curve requires exactly two classes")

    if not np.any(true == positive_label):
        raise ValueError("positive_label must be present in y_true")

    is_positive = true == positive_label
    n_positive = np.count_nonzero(is_positive)
    n_negative = true.size - n_positive

    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_positive = is_positive[order]

    cumulative_tp = np.cumsum(sorted_positive)
    cumulative_fp = np.cumsum(~sorted_positive)

    distinct_indices = np.where(np.diff(sorted_scores) != 0)[0]
    threshold_indices = np.r_[
        distinct_indices,
        sorted_scores.size - 1,
    ]

    tp = cumulative_tp[threshold_indices]
    fp = cumulative_fp[threshold_indices]

    tpr = np.r_[0.0, tp / n_positive]
    fpr = np.r_[0.0, fp / n_negative]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]

    return fpr, tpr, thresholds


def auc(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> float:
    """
    Calculate the area under a curve using the trapezoidal rule.

    Parameters
    ----------
    x : array-like
        Monotonic x-coordinates of the curve.
    y : array-like
        Corresponding y-coordinates.

    Returns
    -------
    float
        Non-negative area under the curve.

    Raises
    ------
    TypeError
        If x or y contains non-real numeric values.
    ValueError
        If the inputs are not one-dimensional, have different lengths,
        contain fewer than two points, contain non-finite values,
        or x is not monotonic.

    Complexity: O(n) time for n curve points.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1D arrays")

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")

    if x_arr.size < 2:
        raise ValueError("at least two points are required")

    if x_arr.dtype.kind not in "iuf" or y_arr.dtype.kind not in "iuf":
        raise TypeError("x and y must contain real numeric values")

    x_arr = x_arr.astype(float, copy=False)
    y_arr = y_arr.astype(float, copy=False)

    if (
        not np.all(np.isfinite(x_arr))
        or not np.all(np.isfinite(y_arr))
    ):
        raise ValueError("x and y must not contain NaN or infinity")

    differences = np.diff(x_arr)

    if not (
        np.all(differences >= 0)
        or np.all(differences <= 0)
    ):
        raise ValueError("x values must be monotonic")

    return float(abs(np.trapezoid(y_arr, x_arr)))


def roc_auc_score(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    positive_label: object = 1,
) -> float:
    """
    Calculate the area under the receiver operating characteristic curve.

    Parameters
    ----------
    y_true : array-like
        Actual binary class labels.
    y_score : array-like
        Numeric scores for the positive class.
    positive_label : object, default=1
        Label treated as the positive class.

    Returns
    -------
    float
        ROC AUC score, generally between 0 and 1.

    Raises
    ------
    TypeError
        If y_score contains non-real numeric values.
    ValueError
        If the ROC curve inputs are invalid or y_true does not contain
        exactly two classes.

    Complexity: O(n log n) time for n prediction scores.
    """
    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
        positive_label=positive_label,
    )

    return auc(fpr, tpr)


class Accuracy:
    def __init__(self, window: int | None = None) -> None:
        self.window = window
        self.reset()

    def reset(self) -> None:
        self._true = []
        self._pred = []

    def update(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> "Accuracy":
        true, pred = _validate_labels(y_true, y_pred)
        self._true.extend(true.tolist())
        self._pred.extend(pred.tolist())
        if self.window is not None:
            self._true = self._true[-self.window:]
            self._pred = self._pred[-self.window:]
        return self

    def result(self) -> float:
        return accuracy(self._true, self._pred) if self._true else 0.0


class ConfusionMatrix:
    def __init__(self, labels: npt.ArrayLike | None = None, window: int | None = None) -> None:
        self.labels = None if labels is None else np.asarray(labels)
        self.window = window
        self.reset()

    def reset(self) -> None:
        self._true = []
        self._pred = []

    def update(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> "ConfusionMatrix":
        true, pred = _validate_labels(y_true, y_pred)
        self._true.extend(true.tolist())
        self._pred.extend(pred.tolist())
        if self.window is not None:
            self._true = self._true[-self.window:]
            self._pred = self._pred[-self.window:]
        return self

    def result(self) -> NDArray[np.int64]:
        if not self._true:
            size = 0 if self.labels is None else len(self.labels)
            return np.zeros((size, size), dtype=np.int64)
        return confusion_matrix(self._true, self._pred, labels=self.labels)


class _StreamingMetric:
    def __init__(self, average: str = "binary", window: int | None = None, positive_label: object = 1) -> None:
        self.average = average
        self.window = window
        self.positive_label = positive_label
        self.reset()

    def reset(self) -> None:
        self._true = []
        self._pred = []

    def update(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> "_StreamingMetric":
        true, pred = _validate_labels(y_true, y_pred)
        self._true.extend(true.tolist())
        self._pred.extend(pred.tolist())
        if self.window is not None:
            self._true = self._true[-self.window:]
            self._pred = self._pred[-self.window:]
        return self

    def _values(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self._true), np.asarray(self._pred)


class Precision(_StreamingMetric):
    def result(self) -> float:
        if not self._true:
            return 0.0
        true, pred = self._values()
        if self.average in ("binary", "default"):
            return precision(true, pred, positive_label=self.positive_label)
        labels = np.unique(np.concatenate((true, pred)))
        values = [precision(true, pred, positive_label=label, zero_division=0) for label in labels]
        if self.average == "weighted":
            weights = [np.count_nonzero(true == label) for label in labels]
            return float(np.average(values, weights=weights))
        return float(np.mean(values))


class Recall(_StreamingMetric):
    def result(self) -> float:
        if not self._true:
            return 0.0
        true, pred = self._values()
        if self.average in ("binary", "default"):
            return recall(true, pred, positive_label=self.positive_label)
        labels = np.unique(np.concatenate((true, pred)))
        values = [recall(true, pred, positive_label=label, zero_division=0) for label in labels]
        if self.average == "weighted":
            weights = [np.count_nonzero(true == label) for label in labels]
            return float(np.average(values, weights=weights))
        return float(np.mean(values))


class F1Score(_StreamingMetric):
    def result(self) -> float:
        if not self._true:
            return 0.0
        true, pred = self._values()
        if self.average in ("binary", "default"):
            return f1(true, pred, positive_label=self.positive_label)
        labels = np.unique(np.concatenate((true, pred)))
        values = [f1(true, pred, positive_label=label, zero_division=0) for label in labels]
        if self.average == "weighted":
            weights = [np.count_nonzero(true == label) for label in labels]
            return float(np.average(values, weights=weights))
        return float(np.mean(values))


class AUC:
    def __init__(self, n_bins: int = 100, positive_label: object = 1) -> None:
        self.n_bins = n_bins
        self.positive_label = positive_label
        self.reset()

    def reset(self) -> None:
        self._true = []
        self._scores = []

    def update(self, y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> "AUC":
        true = np.asarray(y_true)
        scores = np.asarray(y_score)
        if true.ndim != 1 or scores.ndim != 1 or true.shape != scores.shape:
            raise ValueError("y_true and y_score must have the same shapes.")
        if scores.dtype.kind not in "iuf" or not np.all(np.isfinite(scores)):
            raise ValueError("y_score must contain finite values.")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("y_score must be in [0, 1].")
        self._true.extend(true.tolist())
        self._scores.extend(scores.astype(float).tolist())
        return self

    def result(self) -> float:
        true = np.asarray(self._true)
        scores = np.asarray(self._scores)
        positive = true == self.positive_label
        if not np.any(positive) or np.any(~positive):
            pass
        if not np.any(positive) or np.all(positive):
            raise ValueError("AUC requires at least one positive and one negative label.")
        negatives = scores[~positive]
        positives = scores[positive]
        greater = positives[:, None] > negatives[None, :]
        ties = positives[:, None] == negatives[None, :]
        return float((np.count_nonzero(greater) + 0.5 * np.count_nonzero(ties)) / (positives.size * negatives.size))
