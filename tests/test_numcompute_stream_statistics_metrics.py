import numpy as np
import pytest

from numcompute_stream.metrics import (
    AUC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
    confusion_matrix,
    accuracy,
    precision,
    recall,
    f1,
    roc_auc_score,
    roc_curve,
)
from numcompute_stream.stats import StreamingHistogram, StreamingStats, update_stats


def test_streaming_stats_merge_chunks_and_compute_quantiles():
    stats = StreamingStats().update([1.0, 2.0]).update([3.0, 4.0])
    assert stats.result()["mean"] == pytest.approx(2.5)
    assert stats.result()["variance"] == pytest.approx(1.25)
    assert stats.quantile(0.5) == pytest.approx(2.5)
    state = update_stats([1.0, 2.0])
    assert update_stats([3.0], state)["count"] == 3


def test_streaming_stats_rolling_window_discards_older_values():
    stats = StreamingStats(window=3).update([1, 2]).update([3, 4])
    assert stats.result()["count"] == 3
    assert stats.result()["mean"] == pytest.approx(3.0)


def test_streaming_histogram_preserves_edge_counts_after_range_expansion():
    histogram = StreamingHistogram(bins=2).update([0.0, 0.5, 1.0])
    counts_before, edges_before = histogram.result()
    assert counts_before.sum() == 3
    histogram.update([10.0])
    counts_after, edges_after = histogram.result()
    expected, expected_edges = np.histogram([0.0, 0.5, 1.0, 10.0], bins=2, range=(0.0, 10.0))
    np.testing.assert_array_equal(counts_after, expected)
    np.testing.assert_array_equal(edges_after, expected_edges)
    assert edges_before[0] == 0.0


def test_streaming_histogram_one_bin_constant_values_and_sliding_window():
    histogram = StreamingHistogram(bins=1, window=2).update([7.0, 7.0])
    counts, edges = histogram.result()
    assert counts.tolist() == [2]
    assert edges[0] < edges[1]
    histogram.update([9.0])
    assert histogram.result()[0].sum() == 2


def test_streaming_histogram_handles_extreme_finite_range():
    limit = np.finfo(float).max
    counts, edges = StreamingHistogram(bins=2).update([-limit, limit]).result()
    assert counts.sum() == 2
    assert np.all(np.isfinite(edges))


def test_streaming_stats_handles_large_constant_chunks():
    stats = StreamingStats().update([1e308, 1e308]).result()
    assert stats["mean"] == 1e308
    assert stats["variance"] == 0.0
    state = update_stats([1e308, 1e308])
    assert state["mean"] == 1e308
    assert state["variance"] == 0.0


def test_confusion_matrix_handles_non_contiguous_and_explicit_unsorted_labels():
    result = confusion_matrix([10, 40, 20], [40, 40, 10], labels=[40, 10, 20])
    np.testing.assert_array_equal(result, [[1, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_macro_streaming_metrics_support_non_contiguous_labels():
    y_true, y_pred = [10, 20, 40], [10, 40, 40]
    accuracy = Accuracy()
    precision = Precision(average="macro")
    recall = Recall(average="macro")
    f1 = F1Score(average="macro")
    matrix = ConfusionMatrix(labels=[10, 20, 40])
    for metric in (accuracy, precision, recall, f1, matrix):
        metric.update(y_true[:2], y_pred[:2])
        metric.update(y_true[2:], y_pred[2:])
    assert accuracy.result() == pytest.approx(2 / 3)
    assert precision.result() == pytest.approx(0.5)
    assert recall.result() == pytest.approx(2 / 3)
    assert f1.result() == pytest.approx((1 + 0 + 2 / 3) / 3)
    np.testing.assert_array_equal(matrix.result(), [[1, 0, 0], [0, 0, 1], [0, 0, 1]])


def test_streaming_metric_reset_and_rolling_sample_window():
    metric = Accuracy(window=2)
    metric.update([1, 1], [1, 0])
    metric.update([0], [0])
    assert metric.result() == pytest.approx(0.5)
    metric.reset()
    assert metric.result() == 0.0


def test_roc_and_streaming_auc_validate_scores_and_estimate_auc():
    with pytest.raises(ValueError, match="finite"):
        roc_curve([0, 1], [0.1, np.inf])
    auc = AUC(n_bins=100)
    auc.update([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert auc.result() == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        auc.update([0], [2.0])


def test_streaming_precision_recall_reject_shape_mismatch():
    with pytest.raises(ValueError, match="shapes"):
        Precision().update([0, 1], [1])
    with pytest.raises(ValueError, match="shapes"):
        Recall().update([[0]], [[0], [1]])


def test_streaming_stats_rejects_empty_and_nonfinite_chunks():
    with pytest.raises(ValueError, match="empty"):
        StreamingStats().update([])
    with pytest.raises(ValueError, match="finite"):
        StreamingStats().update([1.0, np.nan])


def test_streaming_histogram_window_keeps_only_recent_counts():
    histogram = StreamingHistogram(bins=3, window=3)
    histogram.update([1.0, 2.0]).update([10.0, 20.0])
    counts, _ = histogram.result()

    assert counts.sum() == 3
    assert counts[-1] >= 1


def test_batch_binary_metrics_support_nondefault_positive_label():
    y_true, y_pred = [2, 5, 2, 5], [2, 2, 2, 5]

    assert accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert precision(y_true, y_pred, positive_label=5) == pytest.approx(1.0)
    assert recall(y_true, y_pred, positive_label=5) == pytest.approx(0.5)
    assert f1(y_true, y_pred, positive_label=5) == pytest.approx(2 / 3)


def test_batch_multiclass_metrics_support_micro_and_weighted_averages():
    y_true = [10, 10, 20, 40]
    y_pred = [10, 20, 20, 20]

    assert precision(y_true, y_pred, average="micro") == pytest.approx(0.5)
    assert recall(y_true, y_pred, average="micro") == pytest.approx(0.5)
    assert f1(y_true, y_pred, average="micro") == pytest.approx(0.5)
    assert precision(y_true, y_pred, average="weighted") == pytest.approx(7 / 12)
    assert recall(y_true, y_pred, average="weighted") == pytest.approx(0.5)


def test_batch_metrics_zero_division_policy_is_respected():
    assert precision([1, 0], [0, 0], zero_division=0.25) == pytest.approx(0.25)
    assert recall([0, 0], [1, 1], zero_division=0.25) == pytest.approx(0.25)
    assert f1([1, 0], [0, 0], zero_division=0.25) == pytest.approx(0.0)


def test_batch_metrics_reject_missing_labels_and_absent_positive_label():
    with pytest.raises(ValueError, match="NaN"):
        accuracy([0, np.nan], [0, 1])
    with pytest.raises(ValueError, match="positive_label"):
        precision([0, 0], [0, 0])


def test_confusion_matrix_rejects_labels_that_omit_an_observed_class():
    with pytest.raises(ValueError, match="every observed"):
        confusion_matrix([0, 1], [0, 1], labels=[0])


def test_roc_auc_handles_reversed_scores_and_nondefault_positive_label():
    assert roc_auc_score([2, 5, 2, 5], [0.9, 0.2, 0.8, 0.1], positive_label=5) == pytest.approx(0.0)
    fpr, tpr, thresholds = roc_curve([0, 1], [0.2, 0.8])
    assert fpr[0] == 0.0 and fpr[-1] == 1.0
    assert tpr[0] == 0.0 and tpr[-1] == 1.0
    assert np.isinf(thresholds[0])


def test_streaming_auc_requires_both_classes_and_reset_clears_bins():
    metric = AUC(n_bins=8)
    metric.update([1, 1], [0.7, 0.8])
    with pytest.raises(ValueError, match="positive and one negative"):
        metric.result()
    metric.update([0], [0.2])
    assert metric.result() == pytest.approx(1.0)
    metric.reset()
    with pytest.raises(ValueError, match="positive and one negative"):
        metric.result()


def test_streaming_macro_metrics_can_be_updated_one_sample_at_a_time():
    metric = F1Score(average="macro")
    for truth, prediction in zip([0, 0, 1, 2], [0, 1, 1, 2]):
        metric.update([truth], [prediction])

    assert metric.result() == pytest.approx(7 / 9)


def test_streaming_confusion_matrix_keeps_fixed_explicit_class_order():
    metric = ConfusionMatrix(labels=["survived", "lost"])
    metric.update(["lost", "survived"], ["survived", "survived"])

    np.testing.assert_array_equal(metric.result(), [[1, 0], [1, 0]])


def test_update_stats_preserves_passed_state_object_and_merges_extrema():
    state = update_stats([4.0, 6.0])
    original_id = id(state)
    merged = update_stats([-2.0, 10.0], state)

    assert id(merged) == original_id
    assert merged["count"] == 4
    assert merged["min"] == -2.0
    assert merged["max"] == 10.0
    assert merged["mean"] == pytest.approx(4.5)


def test_streaming_stats_retains_exact_window_quantile():
    stats = StreamingStats(window=4).update([1, 2, 3]).update([10, 20])

    assert stats.result()["count"] == 4
    assert stats.quantile(0.5) == pytest.approx(6.5)


def test_streaming_histogram_rejects_empty_result_before_any_updates():
    with pytest.raises(ValueError, match="No observations"):
        StreamingHistogram(bins=4).result()


@pytest.mark.parametrize("bins", [0, -1, True])
def test_streaming_histogram_rejects_invalid_bin_count(bins):
    with pytest.raises(ValueError, match="bins"):
        StreamingHistogram(bins=bins)


def test_confusion_matrix_accepts_string_labels_and_keeps_requested_order():
    result = confusion_matrix(
        ["lost", "survived", "lost"],
        ["survived", "survived", "lost"],
        labels=["survived", "lost"],
    )
    np.testing.assert_array_equal(result, [[1, 0], [1, 1]])
