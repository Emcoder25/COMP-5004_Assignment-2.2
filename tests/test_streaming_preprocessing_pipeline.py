import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import (
    Imputer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


def test_standard_scaler_partial_fit_matches_combined_statistics():
    scaler = StandardScaler()
    scaler.partial_fit([1.0, 3.0])
    scaler.partial_fit([5.0, 7.0])

    transformed = scaler.transform([1.0, 7.0])
    expected = (np.array([1.0, 7.0]) - 4.0) / np.sqrt(5.0)
    np.testing.assert_allclose(transformed, expected)


def test_standard_scaler_partial_fit_handles_empty_chunk():
    scaler = StandardScaler().partial_fit([2.0, 2.0])
    previous_mean = scaler.mean_

    scaler.partial_fit([])

    assert scaler.mean_ == previous_mean
    assert np.all(np.isfinite(scaler.transform([2.0])))


def test_minmax_scaler_partial_fit_tracks_extrema_across_chunks():
    scaler = MinMaxScaler()
    scaler.partial_fit([2.0, 4.0])
    scaler.partial_fit([-2.0, 6.0])

    np.testing.assert_allclose(scaler.transform([-2.0, 6.0]), [0.0, 1.0])


def test_imputer_partial_fit_updates_mean_and_keeps_missing_values_safe():
    imputer = Imputer(strategy="mean")
    imputer.partial_fit([1.0, np.nan])
    imputer.partial_fit([np.nan, 5.0])

    np.testing.assert_allclose(imputer.transform([np.nan, 1.0, 5.0]), [3.0, 1.0, 5.0])


def test_onehot_encoder_partial_fit_expands_categories_incrementally():
    encoder = OneHotEncoder()
    encoder.partial_fit(["red", "blue"])
    encoder.partial_fit(["green"])

    encoded = encoder.transform(["blue", "green", "red"])
    assert encoded.shape == (3, 3)
    np.testing.assert_array_equal(encoded.sum(axis=1), [1.0, 1.0, 1.0])


def test_onehot_encoder_partial_fit_rejects_mixed_category_type():
    encoder = OneHotEncoder().partial_fit(["red", "blue"])

    with pytest.raises(ValueError, match="string categorical data"):
        encoder.partial_fit([1, 2])


def test_pipeline_partial_fit_updates_columnwise_transformer_and_model():
    class IncrementalModel:
        def partial_fit(self, X, y):
            self.classes_ = np.unique(y)
            self.prediction_ = int(self.classes_[-1])

        def predict(self, X):
            return np.full(len(X), self.prediction_)

    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", IncrementalModel())]
    )
    pipeline.partial_fit([[1.0], [3.0]], [0, 1])
    pipeline.partial_fit([[5.0]], [1])

    assert pipeline.predict([[2.0], [4.0]]).tolist() == [1, 1]
    assert pipeline.named_steps["scale"].mean_ == pytest.approx(3.0)
