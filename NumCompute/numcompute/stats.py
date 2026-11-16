
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def _as_numeric_array(
    data: npt.ArrayLike,
) -> NDArray[np.float64]:
    """
    Convert input to a numeric NumPy array.
    
    Parameters
    ----------
    data : array-like
        Input data containing real numeric values.

    Returns
    -------
    numpy.ndarray
        Floating-point array containing the input data.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.

    Complexity: O(n) time and O(n) space for n input elements.
    """
    arr = np.asarray(data)

    if arr.dtype.kind not in "iuf":
        raise TypeError("Data must contain real numeric values")

    return arr.astype(float, copy=False)


def _validate_data(
    data: npt.ArrayLike,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    """Validate numeric input and optionally allow empty arrays.
    Parameters
    ----------
    data : array-like
        Input data to validate.
    allow_empty : bool, default=False
        Whether to allow an empty array.

    Returns
    -------
    numpy.ndarray
        Validated floating-point array.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the array is empty and allow_empty is False.

    Complexity: O(n) time and O(n) space for n input elements.
    """
    arr = _as_numeric_array(data)

    if arr.size == 0 and not allow_empty:
        raise ValueError("Data must not be empty")

    return arr


def _stable_mean(arr: NDArray[np.float64]) -> float:
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        return 0.0
    return float(np.mean(arr / scale) * scale)


def _stable_variance(arr: NDArray[np.float64]) -> float:
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        return 0.0
    centered = arr / scale - _stable_mean(arr) / scale
    return float(np.mean(centered * centered) * scale * scale)


def mean(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the arithmetic mean.
    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate the mean.
        None calculates the mean over all elements.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Mean value or array of mean values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmean(arr, axis=axis)

    return np.mean(arr, axis=axis)


def median(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the median.
    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate the median.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Median value or array of median values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmedian(arr, axis=axis)

    return np.median(arr, axis=axis)


def std(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Calculate the standard deviation of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate standard deviation.
    ddof : int, default=0
        Delta degrees of freedom. Zero computes population standard
        deviation; one computes sample standard deviation.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Standard deviation value or array of values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty, ddof is invalid, or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")

    if skipna:
        return np.nanstd(arr, axis=axis, ddof=ddof)

    return np.std(arr, axis=axis, ddof=ddof)


def minimum(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Find the minimum value in the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to find the minimum.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Minimum value or array of minimum values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmin(arr, axis=axis)

    return np.min(arr, axis=axis)


def maximum(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Find the maximum value in the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to find the maximum.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Maximum value or array of maximum values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.    
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmax(arr, axis=axis)

    return np.max(arr, axis=axis)


def quantile(
    data: npt.ArrayLike,
    q: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Calculate the requested quantile(s) of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    q : float or array-like
        Quantile value(s) between 0 and 1, inclusive.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate quantiles.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Quantile value or array of quantile values.

    Raises
    ------
    TypeError
        If the input data is non-numeric or complex.
    ValueError
        If the data is empty, q is outside [0, 1], q contains
        non-finite values, or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    arr = _validate_data(data)
    q_arr = np.asarray(q, dtype=float)

    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite values")

    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1")

    if skipna:
        return np.nanquantile(arr, q_arr, axis=axis)

    return np.quantile(arr, q_arr, axis=axis)


def percentile(
    data: npt.ArrayLike,
    q: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Calculate the requested percentile(s) of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    q : float or array-like
        Percentile value(s) between 0 and 100, inclusive.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate percentiles.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Percentile value or array of percentile values.

    Raises
    ------
    TypeError
        If the input data is non-numeric or complex.
    ValueError
        If the data is empty, q is outside [0, 100], q contains
        non-finite values, or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    q_arr = np.asarray(q, dtype=float)

    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite values")

    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100")

    return quantile(data, q_arr / 100.0, axis=axis, skipna=skipna)


def histogram(
    data: npt.ArrayLike,
    bins: int | str | npt.ArrayLike = 10,
    range: tuple[float, float] | None = None,
    density: bool = False,
    skipna: bool = False,
) -> tuple[NDArray[np.int64] | NDArray[np.float64], NDArray[np.float64]]:
    """Calculate a histogram of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    bins : int, str, or array-like, default=10
        Number of bins, a supported binning strategy, or explicit
        bin edges.
    range : tuple of float, optional
        Lower and upper range of the bins.
    density : bool, default=False
        If True, return normalized density values instead of counts.
    skipna : bool, default=False
        If True, ignore NaN values. Infinite values are always rejected.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing histogram counts or densities and bin edges.

    Raises
    ------
    TypeError
        If the data contains non-numeric or complex values.
    ValueError
        If the data has no valid elements, contains NaN without
        skipna=True, contains infinity, or the bin configuration is invalid.

    Complexity: O(n + b) time, where n is the number of observations
    and b is the number of bins.
    """
    arr = _validate_data(data, allow_empty=True).ravel()

    if np.any(np.isinf(arr)):
        raise ValueError("Data must not contain infinite values")

    if skipna:
        arr = arr[~np.isnan(arr)]
    elif np.any(np.isnan(arr)):
        raise ValueError("Data contains NaN; use skipna=True to ignore it")

    if arr.size == 0:
        raise ValueError("Data must contain at least one valid value")

    if isinstance(bins, (int, np.integer)):
        if bins <= 0:
            raise ValueError("Bins must be a positive integer")
    elif not isinstance(bins, str):
        edges = np.asarray(bins, dtype=float)

        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                "Bin edges must be a 1D array with at least 2 values"
            )

        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError(
                "Bin edges must be finite and strictly increasing"
            )

    counts, edges = np.histogram(
        arr,
        bins=bins,
        range=range,
        density=density,
    )

    return counts, edges


def variance(data: npt.ArrayLike) -> float:
    arr = _as_numeric_array(data)
    if arr.size == 0:
        raise ValueError("Data must not be empty")
    return float(np.var(arr))


def update_stats(values: npt.ArrayLike, state: dict[str, object] | None = None, window: int | None = None) -> dict[str, object]:
    arr = _as_numeric_array(values).ravel()
    if arr.size == 0:
        raise ValueError("Data must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Data must contain finite values")

    if state is None:
        state = {"count": 0, "mean": 0.0, "variance": 0.0, "min": float("inf"), "max": float("-inf")}
    combined_window = list(state.get("window", [])) if window is not None else None
    if combined_window is not None:
        combined_window.extend(arr.tolist())
        combined_window = combined_window[-window:]
        arr_window = np.asarray(combined_window, dtype=float)
        state["window"] = combined_window
        state["count"] = int(arr_window.size)
        state["mean"] = _stable_mean(arr_window)
        state["variance"] = _stable_variance(arr_window)
        state["min"] = float(np.min(arr_window))
        state["max"] = float(np.max(arr_window))
        return state

    count = int(state["count"])
    mean = float(state["mean"])
    old_m2 = float(state["variance"]) * count
    new_count = count + arr.size
    batch_mean = _stable_mean(arr)
    batch_m2 = float(np.dot(arr - batch_mean, arr - batch_mean))
    if count == 0:
        new_mean = batch_mean
        new_m2 = batch_m2
    else:
        delta = batch_mean - mean
        new_mean = mean * (count / new_count) + batch_mean * (arr.size / new_count)
        new_m2 = old_m2 + batch_m2 + delta * delta * count * arr.size / new_count
    state.update(
        count=new_count,
        mean=new_mean,
        variance=new_m2 / new_count,
        min=min(float(state["min"]), float(np.min(arr))),
        max=max(float(state["max"]), float(np.max(arr))),
    )
    return state


class StreamingStats:
    def __init__(self, window: int | None = None) -> None:
        self.window = window
        self.reset()

    def reset(self) -> None:
        self._values = []

    def update(self, values: npt.ArrayLike) -> "StreamingStats":
        arr = _as_numeric_array(values).ravel()
        if arr.size == 0:
            raise ValueError("StreamingStats update cannot be empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("StreamingStats values must contain finite values.")
        self._values.extend(arr.tolist())
        if self.window is not None:
            self._values = self._values[-self.window:]
        return self

    def result(self) -> dict[str, object]:
        arr = np.asarray(self._values, dtype=float)
        if arr.size == 0:
            return {"count": 0, "mean": 0.0, "variance": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": int(arr.size),
            "mean": _stable_mean(arr),
            "variance": _stable_variance(arr),
            "std": float(np.sqrt(_stable_variance(arr))),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def quantile(self, q: float) -> float:
        if not self._values:
            raise ValueError("No observations.")
        return float(np.quantile(np.asarray(self._values), q))


class StreamingHistogram:
    def __init__(self, bins: int = 10, window: int | None = None) -> None:
        if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins <= 0:
            raise ValueError("bins must be a positive integer.")
        self.bins = int(bins)
        self.window = window
        self.reset()

    def reset(self) -> None:
        self._values = []

    def update(self, values: npt.ArrayLike) -> "StreamingHistogram":
        arr = _as_numeric_array(values).ravel()
        if arr.size == 0:
            raise ValueError("StreamingHistogram update cannot be empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("StreamingHistogram values must contain finite values.")
        self._values.extend(arr.tolist())
        if self.window is not None:
            self._values = self._values[-self.window:]
        return self

    def result(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        if not self._values:
            raise ValueError("No observations.")
        low = float(np.min(self._values))
        high = float(np.max(self._values))
        if low == high:
            low -= 0.5
            high += 0.5
        scale = max(abs(low), abs(high), 1.0)
        normalized = np.asarray(self._values, dtype=float) / scale
        counts, edges = np.histogram(
            normalized,
            bins=self.bins,
            range=(low / scale, high / scale),
        )
        edge_values = edges * scale
        edge_values[0] = low
        edge_values[-1] = high
        return counts, edge_values
