import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    ColumnSelector,
    Imputer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


def test_standard_scaler_batch_and_incremental_matrix_are_featurewise():
    X = np.array([[1.0, 100.0], [3.0, 200.0], [5.0, 300.0]])
    batch = StandardScaler().fit(X)
    incremental = StandardScaler().partial_fit(X[:1]).partial_fit(X[1:])
    np.testing.assert_allclose(incremental.mean_, batch.mean_)
    np.testing.assert_allclose(incremental.var_, batch.var_)
    np.testing.assert_allclose(incremental.transform(X), batch.transform(X), atol=1e-14)


def test_standard_scaler_one_dimensional_input_and_constant_column():
    scaler = StandardScaler().fit([4.0, 4.0, 4.0])
    np.testing.assert_array_equal(scaler.transform([4.0, 4.0]), [0.0, 0.0])


def test_standard_scaler_handles_large_constant_values_without_overflow():
    scaler = StandardScaler().fit([1e308, 1e308])
    np.testing.assert_array_equal(scaler.transform([1e308]), [0.0])


def test_standard_scaler_ema_updates_and_rejects_bad_alpha():
    scaler = StandardScaler(ema_alpha=0.5).partial_fit([[0.0], [2.0]])
    scaler.partial_fit([[10.0]])
    assert scaler.mean_[0] > 1.0
    with pytest.raises(ValueError, match="ema_alpha"):
        StandardScaler(ema_alpha=0.0)


def test_scaler_rejects_boolean_and_empty_shapes_with_context():
    with pytest.raises(TypeError, match="real numeric"):
        StandardScaler().fit([True, False])
    with pytest.raises(ValueError, match="empty"):
        MinMaxScaler().fit(np.empty((0, 2)))


def test_minmax_scaler_range_validation_clipping_and_extreme_values():
    with pytest.raises(ValueError, match="strictly increasing"):
        MinMaxScaler((2.0, 1.0))
    scaler = MinMaxScaler(clip=True).fit([-1e308, 1e308])
    result = scaler.transform([-1e308, 0.0, 1e308, 1.1e308])
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[[0, 1, 3]], [0.0, 0.5, 1.0])


def test_imputer_supports_per_column_mean_and_configured_missing_marker():
    imputer = Imputer(strategy="mean", missing_values=-999).fit(
        [[1.0, 10.0], [-999.0, 30.0], [3.0, -999.0]]
    )
    np.testing.assert_allclose(
        imputer.transform([[-999.0, -999.0]]), [[2.0, 20.0]]
    )


def test_imputer_supports_categorical_constant_marker():
    imputer = Imputer(strategy="constant", fill_value="unknown", missing_values="?")
    np.testing.assert_array_equal(
        imputer.fit_transform([["red"], ["?"]]), [["red"], ["unknown"]]
    )


def test_encoder_incremental_categories_append_and_preserve_column_meaning():
    encoder = OneHotEncoder(handle_unknown="error")
    encoder.partial_fit(["red", "blue"])
    before = encoder.transform(["red", "blue"])
    old_categories = encoder.categories_[0].copy()
    encoder.partial_fit(["green"])
    after = encoder.transform(["red", "blue", "green"])
    np.testing.assert_array_equal(encoder.categories_[0][:2], old_categories)
    np.testing.assert_array_equal(after[:2, :2], before)
    assert after.shape == (3, 3)


def test_encoder_ignore_unknown_emits_zero_block_and_error_policy_raises():
    ignore = OneHotEncoder(handle_unknown="ignore").fit(["x", "y"])
    np.testing.assert_array_equal(ignore.transform(["z"]), [[0.0, 0.0]])
    with pytest.raises(ValueError, match="Unknown category"):
        OneHotEncoder().fit(["x"]).transform(["y"])


def test_column_selector_indices_and_dtype_routing_for_mixed_object_matrix():
    X = np.array([[1.0, "red", 2.0], [2.0, "blue", 3.0]], dtype=object)
    numeric = ColumnSelector("numeric").fit(X)
    categorical = ColumnSelector("categorical").fit(X)
    assert numeric.transform(X).shape == (2, 2)
    np.testing.assert_array_equal(categorical.transform(X).ravel(), ["red", "blue"])


def test_minmax_incremental_fit_matches_batch_with_constant_feature():
    X = np.array([[2.0, 7.0], [4.0, 7.0], [-2.0, 7.0]])
    batch = MinMaxScaler().fit(X)
    incremental = MinMaxScaler().partial_fit(X[:1]).partial_fit(X[1:])

    np.testing.assert_allclose(incremental.transform(X), batch.transform(X))
    np.testing.assert_array_equal(incremental.transform(X)[:, 1], [0.0, 0.0, 0.0])


def test_minmax_scaler_rejects_transform_feature_count_mismatch():
    scaler = MinMaxScaler().fit([[0.0, 1.0], [1.0, 2.0]])

    with pytest.raises(ValueError, match="feature count"):
        scaler.transform([[1.0]])


def test_imputer_median_strategy_updates_across_chunks():
    imputer = Imputer(strategy="median")
    imputer.partial_fit([[1.0], [np.nan]])
    imputer.partial_fit([[5.0], [9.0]])

    np.testing.assert_allclose(imputer.transform([[np.nan]]), [[5.0]])


def test_imputer_rejects_transform_with_wrong_number_of_columns():
    imputer = Imputer(strategy="constant", fill_value=0).fit([[1.0, 2.0]])

    with pytest.raises(ValueError, match="feature count"):
        imputer.transform([[np.nan]])


def test_onehot_encoder_multicolumn_layout_and_empty_transform():
    encoder = OneHotEncoder().fit([["male", "S"], ["female", "C"]])
    encoded = encoder.transform([["female", "S"], ["male", "C"]])

    assert encoded.shape == (2, 4)
    np.testing.assert_array_equal(encoded.sum(axis=1), [2.0, 2.0])
    assert encoder.transform(np.empty((0, 2), dtype=str)).shape == (0, 4)


def test_onehot_encoder_rejects_missing_values_and_changed_feature_count():
    with pytest.raises(ValueError, match="missing"):
        OneHotEncoder().fit(["red", None])
    encoder = OneHotEncoder().fit([["red", "S"]])
    with pytest.raises(ValueError, match="feature count"):
        encoder.transform([["red"]])


def test_column_selector_supports_named_columns_and_checks_names():
    X = np.array([[1.0, "red"], [2.0, "blue"]], dtype=object)
    selector = ColumnSelector(["port"], feature_names=["age", "port"]).fit(X)
    np.testing.assert_array_equal(selector.transform(X).ravel(), ["red", "blue"])

    with pytest.raises(ValueError, match="feature_names length"):
        ColumnSelector([0], feature_names=["age"]).fit(X)


def test_transformers_report_unfitted_state_before_transform():
    with pytest.raises(ValueError, match="not fitted"):
        StandardScaler().transform([1.0])
    with pytest.raises(ValueError, match="not fitted"):
        MinMaxScaler().transform([1.0])
    with pytest.raises(ValueError, match="not fitted"):
        OneHotEncoder().transform(["red"])


def test_standard_scaler_rejects_nonfinite_values_without_mutating_previous_state():
    scaler = StandardScaler().partial_fit([[1.0], [3.0]])
    previous_mean = scaler.mean_.copy()

    with pytest.raises(ValueError, match="finite"):
        scaler.partial_fit([[np.inf]])

    np.testing.assert_array_equal(scaler.mean_, previous_mean)


def test_standard_scaler_rejects_feature_width_change():
    scaler = StandardScaler().fit([[1.0, 2.0]])

    with pytest.raises(ValueError, match="feature count changed"):
        scaler.partial_fit([[3.0]])


def test_imputer_preserves_nonmissing_categorical_values():
    imputer = Imputer(strategy="constant", fill_value="Unknown").fit(
        [["female", "S"], ["male", ""]]
    )
    transformed = imputer.transform([["female", "C"], ["male", ""]])

    np.testing.assert_array_equal(transformed, [["female", "C"], ["male", "Unknown"]])


def test_column_selector_rejects_out_of_range_columns():
    with pytest.raises(ValueError, match="Invalid selected columns"):
        ColumnSelector([2]).fit([[1.0, 2.0]])
