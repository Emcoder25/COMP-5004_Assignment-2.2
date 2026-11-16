"""Compatibility preprocessing transformers for streaming workflows."""

from numcompute.preprocessing import (
    ColumnSelector,
    Imputer,
    MinMaxScaler as _MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


class MinMaxScaler(_MinMaxScaler):
    """Min-max scaler that supports feature matrices for streaming data."""

    def __init__(
        self,
        feature_range: tuple[float, float] = (0.0, 1.0),
        clip: bool = False,
    ) -> None:
        super().__init__(feature_range=feature_range, clip=clip)
        self._allow_2d = True


__all__ = [
    "ColumnSelector",
    "Imputer",
    "MinMaxScaler",
    "OneHotEncoder",
    "StandardScaler",
]
