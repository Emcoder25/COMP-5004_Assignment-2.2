"""NumPy preprocessing transformers with batch and incremental support."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Any


def _shape(X: npt.ArrayLike) -> tuple[np.ndarray, bool]:
    arr = np.asarray(X)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    was_1d = arr.ndim == 1
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 1D or 2D array. Received shape {arr.shape}.")
    return arr, was_1d


def _stable_mean(arr: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(arr), axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return np.mean(arr / scale, axis=0) * scale


def _category_kind(values: np.ndarray) -> str:
    items = values.tolist()
    if any(value is None for value in items):
        raise ValueError("OneHotEncoder cannot encode missing values.")
    is_string = [isinstance(value, (str, np.str_)) for value in items]
    is_integer = [
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        for value in items
    ]
    if any(is_string) and any(is_integer):
        raise ValueError("OneHotEncoder cannot mix string and numeric categorical data.")
    if all(is_string):
        return "string"
    if all(is_integer):
        if any(value < 0 for value in items):
            raise ValueError("OneHotEncoder requires non-negative integers.")
        return "integer"
    raise ValueError("OneHotEncoder expects integer categorical data or strings.")


class StandardScaler:
    def __init__(self, epsilon: float = 1e-8, ema_alpha: float | None = None) -> None:
        if ema_alpha is not None and (isinstance(ema_alpha, bool) or not 0 < float(ema_alpha) <= 1):
            raise ValueError("ema_alpha must be in (0, 1].")
        self.epsilon = epsilon
        self.ema_alpha = ema_alpha

    def fit(self, X: npt.ArrayLike) -> "StandardScaler":
        arr, one_d = _shape(X)
        if arr.size == 0:
            raise ValueError("StandardScaler cannot fit on an empty array.")
        if arr.dtype.kind not in "iuf" or np.issubdtype(arr.dtype, np.bool_):
            raise TypeError("StandardScaler requires real numeric values.")
        arr = arr.astype(float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("StandardScaler values must be finite.")
        means = _stable_mean(arr)
        centered = arr - means
        vars_ = np.mean(centered * centered, axis=0).astype(float)
        scales = np.sqrt(vars_)
        scales = np.where(scales == 0, 1.0, scales)
        self.mean_, self.var_, self.scale_ = (means[0] if one_d else means,
                                               vars_[0] if one_d else vars_,
                                               scales[0] if one_d else scales)
        self.n_samples_seen_ = int(arr.shape[0])
        self.n_features_in_ = arr.shape[1]
        self._one_d = one_d
        return self

    def partial_fit(self, X: npt.ArrayLike) -> "StandardScaler":
        arr, one_d = _shape(X)
        if arr.shape[0] == 0:
            if hasattr(self, "mean_"):
                return self
            raise ValueError("StandardScaler cannot fit on an empty array.")
        if arr.dtype.kind not in "iuf" or np.issubdtype(arr.dtype, np.bool_):
            raise TypeError("StandardScaler requires real numeric values.")
        arr = arr.astype(float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("StandardScaler values must be finite.")
        if hasattr(self, "n_features_in_") and arr.shape[1] != self.n_features_in_:
            raise ValueError("StandardScaler feature count changed.")
        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = 0
            self.n_features_in_ = arr.shape[1]
            self.mean_ = np.zeros(arr.shape[1])
            self.var_ = np.zeros(arr.shape[1])
        old_count = self.n_samples_seen_
        batch_mean = _stable_mean(arr)
        centered = arr - batch_mean
        batch_var = np.mean(centered * centered, axis=0).astype(float)
        new_count = old_count + arr.shape[0]
        if old_count:
            old_mean = np.asarray(self.mean_, dtype=float).reshape(-1)
            old_var = np.asarray(self.var_, dtype=float).reshape(-1)
            if self.ema_alpha is None:
                delta = batch_mean - old_mean
                combined_var = (old_var * old_count + batch_var * arr.shape[0]
                                + delta * delta * old_count * arr.shape[0] / new_count) / new_count
                combined_mean = old_mean + delta * arr.shape[0] / new_count
            else:
                combined_mean = (1 - self.ema_alpha) * old_mean + self.ema_alpha * batch_mean
                combined_var = (1 - self.ema_alpha) * old_var + self.ema_alpha * batch_var
        else:
            combined_mean, combined_var = batch_mean, batch_var
        mean_out = combined_mean[0] if one_d else combined_mean
        var_out = combined_var[0] if one_d else combined_var
        if one_d:
            self.mean_, self.var_ = mean_out, var_out
            self.scale_ = 1.0 if var_out == 0 else float(np.sqrt(var_out))
        else:
            self.mean_, self.var_ = combined_mean, combined_var
            self.scale_ = np.where(combined_var == 0, 1.0, np.sqrt(combined_var))
        self.n_samples_seen_ = new_count
        self.n_features_in_ = arr.shape[1]
        self._one_d = one_d
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise ValueError("This StandardScaler instance is not fitted yet.")
        arr, one_d = _shape(X)
        if arr.dtype.kind in "US":
            arr = arr.astype(object)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError("StandardScaler feature count changed.")
        if arr.dtype.kind not in "iuf":
            raise TypeError("StandardScaler requires real numeric values.")
        arr = arr.astype(float)
        mean = np.asarray(self.mean_, dtype=float).reshape(-1)
        scale = np.asarray(self.scale_, dtype=float).reshape(-1)
        result = (arr - mean) / scale
        return result.ravel() if one_d else result

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScaler:
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), clip: bool = False) -> None:
        if len(feature_range) != 2 or feature_range[0] >= feature_range[1]:
            raise ValueError("MinMax feature_range must be strictly increasing.")
        self.feature_range = feature_range
        self.clip = clip

    def fit(self, X: npt.ArrayLike) -> "MinMaxScaler":
        arr, one_d = _shape(X)
        if not one_d and not getattr(self, "_allow_2d", False):
            raise ValueError("X must be a 1D array.")
        if arr.size == 0:
            raise ValueError("MinMaxScaler cannot fit on an empty array.")
        if arr.dtype.kind not in "iuf":
            raise TypeError("MinMaxScaler requires real numeric values.")
        arr = arr.astype(float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("MinMaxScaler values must be finite.")
        mins = np.min(arr, axis=0)
        maxs = np.max(arr, axis=0)
        self.data_min_, self.data_max_ = (mins[0] if one_d else mins, maxs[0] if one_d else maxs)
        self.n_features_in_ = arr.shape[1]
        self._one_d = one_d
        with np.errstate(over="ignore"):
            ranges = maxs - mins
        ranges = np.where(ranges == 0, 1.0, ranges)
        self.data_range_ = float(ranges[0]) if one_d else ranges
        return self

    def partial_fit(self, X: npt.ArrayLike) -> "MinMaxScaler":
        arr, one_d = _shape(X)
        if not one_d and not getattr(self, "_allow_2d", False):
            raise ValueError("X must be a 1D array.")
        if arr.size == 0:
            if hasattr(self, "data_min_"):
                return self
            raise ValueError("MinMaxScaler cannot fit on an empty array.")
        if arr.dtype.kind not in "iuf":
            raise ValueError("MinMaxScaler values must be finite real values.")
        arr = arr.astype(float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("MinMaxScaler values must be finite real values.")
        mins = np.min(arr, axis=0)
        maxs = np.max(arr, axis=0)
        if hasattr(self, "data_min_"):
            old_mins = np.asarray(self.data_min_, dtype=float).reshape(-1)
            old_maxs = np.asarray(self.data_max_, dtype=float).reshape(-1)
            if old_mins.size != arr.shape[1]:
                raise ValueError("feature count changed.")
            mins, maxs = np.minimum(old_mins, mins), np.maximum(old_maxs, maxs)
        self.data_min_, self.data_max_ = (mins[0] if one_d else mins, maxs[0] if one_d else maxs)
        self.n_features_in_ = arr.shape[1]
        self._one_d = one_d
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        if not hasattr(self, "data_min_"):
            raise ValueError("This MinMaxScaler instance is not fitted yet.")
        arr, one_d = _shape(X)
        if not one_d and not getattr(self, "_allow_2d", False):
            raise ValueError("X must be a 1D array.")
        if arr.shape[1] != self.n_features_in_:
            raise ValueError("MinMaxScaler transform feature count mismatch.")
        arr = arr.astype(float)
        mins = np.asarray(self.data_min_, dtype=float).reshape(-1)
        maxs = np.asarray(self.data_max_, dtype=float).reshape(-1)
        low, high = self.feature_range
        scale = np.maximum(np.maximum(np.abs(mins), np.abs(maxs)), 1.0)
        scaled_arr = arr / scale
        scaled_mins = mins / scale
        scaled_maxs = maxs / scale
        spans = scaled_maxs - scaled_mins
        spans = np.where(spans == 0, 1.0, spans)
        result = (scaled_arr - scaled_mins) / spans * (high - low) + low
        if self.clip:
            result = np.clip(result, low, high)
        return result.ravel() if one_d else result

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)


class Imputer:
    def __init__(self, strategy: str = "mean", fill_value: Any = 0.0, missing_values: Any = np.nan) -> None:
        if strategy not in {"mean", "median", "constant"}:
            raise ValueError("Strategy must be 'mean', 'median', or 'constant'")
        self.strategy, self.fill_value, self.missing_values = strategy, fill_value, missing_values

    def _mask(self, arr: np.ndarray) -> np.ndarray:
        if isinstance(self.missing_values, (float, np.floating)) and np.isnan(self.missing_values):
            if arr.dtype.kind in "OUS":
                return np.fromiter((value is None or (isinstance(value, str) and value.strip().lower() in {"", "nan", "none"})
                                    or (isinstance(value, (float, np.floating)) and np.isnan(value))
                                    for value in arr.ravel()), dtype=bool).reshape(arr.shape)
            return np.isnan(arr) if arr.dtype.kind in "fc" else np.zeros(arr.shape, dtype=bool)
        return arr == self.missing_values

    def fit(self, X: npt.ArrayLike) -> "Imputer":
        arr, one_d = _shape(X)
        if arr.dtype.kind in "US":
            arr = arr.astype(object)
        if arr.size == 0:
            raise ValueError("Imputer cannot fit on an empty array.")
        mask = self._mask(arr)
        stats = []
        for column in range(arr.shape[1]):
            values = arr[:, column][~mask[:, column]]
            if self.strategy == "constant":
                stats.append(self.fill_value)
            elif values.size == 0:
                stats.append(self.fill_value)
            else:
                try:
                    numeric = values.astype(float)
                    stats.append(float(np.mean(numeric)) if self.strategy == "mean" else float(np.median(numeric)))
                except (TypeError, ValueError):
                    if not isinstance(self.fill_value, (int, float, np.number)):
                        stats.append(self.fill_value)
                    else:
                        raise ValueError(f"Cannot compute strategy='{self.strategy}' on non-numeric string data.")
        self.statistics_ = np.asarray(stats, dtype=object)
        self.n_features_in_ = arr.shape[1]
        self._one_d = one_d
        self._observed_values = [
            arr[:, column][~mask[:, column]].tolist()
            for column in range(arr.shape[1])
        ]
        return self

    def partial_fit(self, X: npt.ArrayLike) -> "Imputer":
        arr, one_d = _shape(X)
        if arr.size == 0:
            return self
        if not hasattr(self, "n_features_in_"):
            return self.fit(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError("Imputer feature count changed.")
        mask = self._mask(arr)
        for column in range(arr.shape[1]):
            values = arr[:, column][~mask[:, column]]
            if values.size == 0 or self.strategy == "constant":
                continue
            try:
                numeric = values.astype(float)
                if self.strategy == "mean":
                    old = getattr(self, "_observed_values", [[] for _ in range(arr.shape[1])])
                    combined = old[column] + numeric.tolist()
                    self._observed_values[column] = combined
                    statistic = float(np.mean(combined))
                else:
                    old = getattr(self, "_observed_values", [[] for _ in range(arr.shape[1])])
                    combined = old[column] + numeric.tolist()
                    self._observed_values[column] = combined
                    statistic = float(np.median(combined))
                self.statistics_[column] = statistic
            except (TypeError, ValueError):
                pass
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        if not hasattr(self, "statistics_"):
            raise ValueError("Imputer is not fitted.")
        arr, one_d = _shape(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError("Imputer transform feature count mismatch.")
        if arr.dtype.kind in "US":
            arr = arr.astype(object)
        result = arr.copy()
        mask = self._mask(result)
        for column in range(result.shape[1]):
            result[mask[:, column], column] = self.statistics_[column]
        return result.ravel() if one_d else result

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)


class OneHotEncoder:
    def __init__(self, handle_unknown: str = "error") -> None:
        if handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be error or ignore.")
        self.handle_unknown = handle_unknown

    def fit(self, X: npt.ArrayLike) -> "OneHotEncoder":
        arr, one_d = _shape(X)
        if arr.dtype.kind in "OUS":
            arr = arr.astype(object)
        if arr.size == 0:
            raise ValueError("OneHotEncoder cannot fit empty data.")
        categories = []
        category_types = []
        for column in range(arr.shape[1]):
            values = arr[:, column]
            kind = _category_kind(values)
            categories.append(np.unique(values.astype(object)))
            category_types.append(kind)
        self.categories_ = categories
        self.unique_categories_ = categories[0]
        self.category_types_ = category_types
        self.n_features_in_ = arr.shape[1]
        self._one_d = one_d
        return self

    def partial_fit(self, X: npt.ArrayLike) -> "OneHotEncoder":
        arr, one_d = _shape(X)
        if arr.dtype.kind in "OUS":
            arr = arr.astype(object)
        if arr.size == 0:
            return self
        if hasattr(self, "n_features_in_") and arr.shape[1] != self.n_features_in_:
            raise ValueError("OneHotEncoder feature count changed.")
        kinds = []
        for column in range(arr.shape[1]):
            kinds.append(_category_kind(arr[:, column]))
        if hasattr(self, "category_types_") and any(old != new for old, new in zip(self.category_types_, kinds)):
            raise ValueError(f"OneHotEncoder expects {self.category_types_[0]} categorical data.")
        if not hasattr(self, "categories_"):
            return self.fit(X)
        for column in range(arr.shape[1]):
            existing = self.categories_[column].tolist()
            additions = [
                value for value in arr[:, column].tolist()
                if value not in existing
            ]
            self.categories_[column] = np.asarray(existing + additions, dtype=object)
        self.unique_categories_ = self.categories_[0]
        self.category_types_ = kinds
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        if not hasattr(self, "categories_"):
            raise ValueError("OneHotEncoder is not fitted.")
        arr, one_d = _shape(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError("OneHotEncoder transform feature count mismatch.")
        category_sets = self.categories_ if not one_d else [self.categories_] if isinstance(self.categories_, np.ndarray) else self.categories_
        if not isinstance(category_sets, list):
            category_sets = [category_sets]
        blocks = []
        for column, categories in enumerate(category_sets):
            expected_kind = self.category_types_[column]
            for value in arr[:, column]:
                kind = _category_kind(np.asarray([value], dtype=object))
                if kind != expected_kind:
                    raise ValueError(f"OneHotEncoder expects {expected_kind} categorical data.")
            block = np.zeros((arr.shape[0], len(categories)), dtype=float)
            for row, value in enumerate(arr[:, column]):
                matches = np.flatnonzero(np.asarray(categories) == value)
                if matches.size:
                    block[row, int(matches[0])] = 1.0
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category {value!r}; unknown categories are not allowed.")
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)


class ColumnSelector:
    def __init__(self, columns: npt.ArrayLike | str, feature_names: npt.ArrayLike | None = None) -> None:
        self.columns, self.feature_names = columns, feature_names

    def fit(self, X: npt.ArrayLike) -> "ColumnSelector":
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("ColumnSelector requires a two-dimensional array.")
        if self.feature_names is not None and len(self.feature_names) != arr.shape[1]:
            raise ValueError("feature_names length must match input feature count.")
        if isinstance(self.columns, str):
            if self.columns == "numeric":
                selected = [
                    index for index in range(arr.shape[1])
                    if all(
                        isinstance(value, (int, float, np.integer, np.floating))
                        and not isinstance(value, (bool, np.bool_))
                        for value in arr[:, index].tolist()
                    )
                ]
            elif self.columns == "categorical":
                numeric = {
                    index for index in range(arr.shape[1])
                    if all(
                        isinstance(value, (int, float, np.integer, np.floating))
                        and not isinstance(value, (bool, np.bool_))
                        for value in arr[:, index].tolist()
                    )
                }
                selected = [index for index in range(arr.shape[1]) if index not in numeric]
            else:
                raise ValueError("Invalid selected columns.")
        else:
            selected = list(np.asarray(self.columns).ravel())
            if self.feature_names is not None:
                names = {name: index for index, name in enumerate(self.feature_names)}
                selected = [names.get(column, column) for column in selected]
            selected = [int(column) for column in selected]
        if any(column < 0 or column >= arr.shape[1] for column in selected):
            raise ValueError("Invalid selected columns.")
        self.selected_ = np.asarray(selected, dtype=int)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(X)
        if not hasattr(self, "selected_"):
            raise ValueError("ColumnSelector is not fitted.")
        if arr.ndim != 2:
            raise ValueError("ColumnSelector requires a two-dimensional array.")
        if arr.shape[1] != self.n_features_in_:
            raise ValueError("ColumnSelector input feature count changed.")
        return arr[:, self.selected_]

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)
