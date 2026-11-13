import numpy as np
import pytest

from numcompute.metrics import (
    Accuracy,
    AUC,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)
from numcompute.stats import update_stats, variance
from numcompute.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


def test_streaming_accuracy_accumulates_weighted_counts_across_chunks():
    metric = Accuracy()
    metric.update([1, 0, 1], [1, 1, 1])
    metric.update([0], [0])

    assert metric.result() == pytest.approx(0.75)


def test_streaming_accuracy_reset_clears_state():
    metric = Accuracy()
    metric.update([1, 0], [1, 1])
    metric.reset()

    assert metric.result() == 0.0


def test_streaming_precision_recall_and_f1_accumulate_confusion_counts():
    y_true = [1, 0, 1]
    y_pred = [1, 1, 0]
    precision = Precision()
    recall = Recall()
    f1 = F1Score()

    for metric in (precision, recall, f1):
        metric.update(y_true[:2], y_pred[:2])
        metric.update(y_true[2:], y_pred[2:])

    assert precision.result() == pytest.approx(0.5)
    assert recall.result() == pytest.approx(0.5)
    assert f1.result() == pytest.approx(0.5)


def test_streaming_confusion_matrix_accumulates_multiple_chunks():
    metric = ConfusionMatrix(labels=[0, 1])
    metric.update([0, 1], [0, 1])
    metric.update([0, 1], [1, 0])

    np.testing.assert_array_equal(metric.result(), [[1, 1], [1, 1]])


def test_streaming_auc_combines_scores_across_chunks():
    metric = AUC()
    metric.update([0, 1], [0.1, 0.9])
    metric.update([0, 1], [0.2, 0.8])

    assert metric.result() == pytest.approx(1.0)


def test_streaming_metrics_reject_mismatched_chunk_shapes():
    metric = Accuracy()
    with pytest.raises(ValueError, match="same shape"):
        metric.update([0, 1], [0])


def test_update_stats_merges_chunked_mean_variance_and_count():
    state = update_stats([1.0, 2.0])
    state = update_stats([3.0, 4.0], state)

    assert state["count"] == 4
    assert state["mean"] == pytest.approx(2.5)
    assert state["variance"] == pytest.approx(1.25)
    assert variance([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.25)


def test_update_stats_supports_rolling_window():
    state = update_stats([1.0, 2.0], window=3)
    state = update_stats([3.0, 4.0], state, window=3)

    assert state["window"] == [2.0, 3.0, 4.0]


def test_visualisation_functions_return_figure_and_axes():
    for figure, axes in (
        plot_metric_over_time([0.4, 0.8], "Accuracy", "score"),
        compare_models([0.4, 0.8], [0.5, 0.9], ["small", "large"]),
        plot_predictions_vs_ground_truth([0, 1], [0, 1]),
    ):
        assert figure is not None
        assert axes is not None
