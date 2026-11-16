import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream.io import iter_csv, load_csv
from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


def test_load_csv_infers_mixed_numeric_and_text_columns(tmp_path):
    path = tmp_path / "passengers.csv"
    path.write_text("age,port\n22.5,S\n38.0,C\n", encoding="utf-8")

    header, data = load_csv(path)

    assert header == ["age", "port"]
    assert data.shape == (2, 2)
    assert data.dtype == object
    assert data[:, 0].tolist() == [22.5, 38.0]
    assert data[:, 1].tolist() == ["S", "C"]


def test_iter_csv_yields_bounded_chunks_and_same_header(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("x,label\n1,a\n2,b\n3,c\n4,d\n5,e\n", encoding="utf-8")

    chunks = list(iter_csv(path, chunksize=2))

    assert [header for header, _ in chunks] == [["x", "label"]] * 3
    assert [len(rows) for _, rows in chunks] == [2, 2, 1]
    np.testing.assert_array_equal(np.concatenate([rows[:, 0] for _, rows in chunks]), [1, 2, 3, 4, 5])


def test_csv_fill_and_drop_missing_records(tmp_path):
    path = tmp_path / "missing.csv"
    path.write_text("name,age\nAda,36\nBob,\n", encoding="utf-8")

    _, filled = load_csv(path, fill_value="Unknown")
    _, dropped = load_csv(path, missing_strategy="drop")

    assert filled[1, 1] == "Unknown"
    assert dropped.shape == (1, 2)
    assert dropped[0, 0] == "Ada"


@pytest.mark.parametrize(
    ("chunksize", "message"),
    [(0, "positive"), (True, "positive integer")],
)
def test_iter_csv_rejects_invalid_chunk_sizes(tmp_path, chunksize, message):
    path = tmp_path / "rows.csv"
    path.write_text("x\n1\n", encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        list(iter_csv(path, chunksize=chunksize))


def test_plot_metric_over_time_sets_titles_labels_and_x_coordinates():
    fig, ax = plot_metric_over_time([0.25, 0.75], title="Accuracy", ylabel="score")
    try:
        assert ax.get_title() == "Accuracy"
        assert ax.get_xlabel() == "Chunk"
        assert ax.get_ylabel() == "score"
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.25, 0.75])
    finally:
        plt.close(fig)


def test_plot_metric_over_time_saves_requested_file(tmp_path):
    image_path = tmp_path / "metric.png"
    fig, _ = plot_metric_over_time([1.0], save_path=str(image_path))
    plt.close(fig)

    assert image_path.exists()
    assert image_path.stat().st_size > 0


def test_plot_metric_over_time_show_flag_calls_matplotlib(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    fig, _ = plot_metric_over_time([0.5], show=True)
    plt.close(fig)

    assert shown == [True]


def test_plot_metric_over_time_handles_empty_history():
    fig, ax = plot_metric_over_time([])
    try:
        assert ax.lines[0].get_xdata().size == 0
        assert ax.lines[0].get_ydata().size == 0
    finally:
        plt.close(fig)


def test_plot_metric_over_time_keeps_float32_values_and_custom_labels():
    values = np.array([0.125, 0.875], dtype=np.float32)
    fig, ax = plot_metric_over_time(values, title="Rolling score", ylabel="F1")
    try:
        np.testing.assert_allclose(ax.lines[0].get_ydata(), values)
        assert ax.get_title() == "Rolling score"
        assert ax.get_ylabel() == "F1"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("values", [[[1.0]], [[1.0, 2.0], [3.0, 4.0]]])
def test_plot_metric_over_time_rejects_non_vector_input(values):
    with pytest.raises(ValueError, match="1D"):
        plot_metric_over_time(values)


@pytest.mark.parametrize(
    ("values", "error"),
    [(["not-a-number"], ValueError), ([[1.0], [2.0]], ValueError)],
)
def test_plot_metric_over_time_rejects_unconvertible_or_multidimensional_data(values, error):
    with pytest.raises(error):
        plot_metric_over_time(values)


def test_compare_models_plots_two_histories_and_legend_labels():
    fig, ax = compare_models([0.5, 0.6], [0.4, 0.7], ["baseline", "updated"])
    try:
        assert [line.get_label() for line in ax.lines] == ["baseline", "updated"]
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1])
        assert ax.get_legend() is not None
    finally:
        plt.close(fig)


def test_compare_models_uses_requested_title_and_metric_label():
    fig, ax = compare_models(
        [0.2, 0.4],
        [0.3, 0.5],
        ["tree A", "tree B"],
        title="Titanic survival",
        ylabel="Accuracy",
    )
    try:
        assert ax.get_title() == "Titanic survival"
        assert ax.get_ylabel() == "Accuracy"
        assert ax.get_xlabel() == "Chunk"
    finally:
        plt.close(fig)


def test_compare_models_accepts_empty_histories():
    fig, ax = compare_models([], [], ["first", "second"])
    try:
        assert len(ax.lines) == 2
        assert all(line.get_xdata().size == 0 for line in ax.lines)
    finally:
        plt.close(fig)


def test_compare_models_rejects_non_numeric_values():
    with pytest.raises(ValueError):
        compare_models(["bad"], [0.5], ["first", "second"])


def test_compare_models_saves_figure_and_calls_show(tmp_path, monkeypatch):
    image_path = tmp_path / "comparison.png"
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    fig, _ = compare_models([0.1], [0.2], ["a", "b"], save_path=str(image_path), show=True)
    plt.close(fig)

    assert image_path.exists()
    assert image_path.stat().st_size > 0
    assert shown == [True]


@pytest.mark.parametrize(
    ("first", "second", "labels", "message"),
    [
        ([1], [1, 2], ["a", "b"], "equal-length"),
        ([[1]], [[1]], ["a", "b"], "1D"),
        ([1], [1], ["only-one"], "exactly two"),
    ],
)
def test_compare_models_rejects_bad_histories_or_labels(first, second, labels, message):
    with pytest.raises(ValueError, match=message):
        compare_models(first, second, labels)


def test_plot_predictions_displays_values_and_rejects_shape_mismatch():
    fig, ax = plot_predictions_vs_ground_truth([0, 1, 1], [0, 0, 1], title="Survival")
    try:
        assert ax.get_title() == "Survival"
        assert [line.get_label() for line in ax.lines] == ["Ground truth", "Predictions"]
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), [0, 0, 1])
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="equal-length 1D"):
        plot_predictions_vs_ground_truth([0, 1], [1])


def test_plot_predictions_handles_string_class_labels():
    fig, ax = plot_predictions_vs_ground_truth(
        ["no", "yes", "yes"], ["no", "no", "yes"]
    )
    try:
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), ["no", "yes", "yes"])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), ["no", "no", "yes"])
    finally:
        plt.close(fig)


def test_plot_predictions_accepts_integer_arrays_without_mutating_inputs():
    truth = np.array([0, 1, 1], dtype=np.int64)
    predictions = np.array([1, 1, 0], dtype=np.int64)
    truth_before, predictions_before = truth.copy(), predictions.copy()
    fig, _ = plot_predictions_vs_ground_truth(truth, predictions)
    plt.close(fig)

    np.testing.assert_array_equal(truth, truth_before)
    np.testing.assert_array_equal(predictions, predictions_before)


def test_plot_predictions_saves_figure_and_calls_show(tmp_path, monkeypatch):
    image_path = tmp_path / "predictions.png"
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    fig, _ = plot_predictions_vs_ground_truth(
        [0, 1], [1, 1], save_path=str(image_path), show=True
    )
    plt.close(fig)

    assert image_path.exists()
    assert image_path.stat().st_size > 0
    assert shown == [True]


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([[0], [1]], [0, 1]),
        ([0, 1], [[0, 1]]),
        ([], [0]),
    ],
)
def test_plot_predictions_rejects_non_vector_inputs(y_true, y_pred):
    with pytest.raises(ValueError, match="equal-length 1D"):
        plot_predictions_vs_ground_truth(y_true, y_pred)


def test_all_plot_functions_return_distinct_axes_and_figures():
    results = [
        plot_metric_over_time([0.1, 0.9]),
        compare_models([0.1, 0.9], [0.2, 0.8], ["a", "b"]),
        plot_predictions_vs_ground_truth([0, 1], [0, 0]),
    ]
    try:
        assert len({id(fig) for fig, _ in results}) == 3
        assert len({id(ax) for _, ax in results}) == 3
    finally:
        for fig, _ in results:
            plt.close(fig)


def test_iter_csv_supports_custom_delimiter_and_explicit_dtype(tmp_path):
    path = tmp_path / "semicolon.csv"
    path.write_text("x;label\n1;2\n3;4\n", encoding="utf-8")

    header, data = load_csv(path, delimiter=";", dtype=int)
    chunks = list(iter_csv(path, chunksize=1, delimiter=";", dtype=int))

    assert header == ["x", "label"]
    assert data.dtype.kind in "iu"
    np.testing.assert_array_equal(data, [[1, 2], [3, 4]])
    assert [len(rows) for _, rows in chunks] == [1, 1]


def test_csv_parses_quoted_delimiters_and_header_only_files(tmp_path):
    quoted_path = tmp_path / "quoted.csv"
    quoted_path.write_text('name,comment\nAda,"likes, tea"\n', encoding="utf-8")
    header_path = tmp_path / "header.csv"
    header_path.write_text("name,age\n", encoding="utf-8")

    header, quoted = load_csv(quoted_path)
    empty_header, empty = load_csv(header_path)

    assert header == ["name", "comment"]
    assert quoted[0, 1] == "likes, tea"
    assert empty_header == ["name", "age"]
    assert empty.shape == (0, 2)


def test_csv_rejects_invalid_delimiter_missing_mode_and_dtype(tmp_path):
    path = tmp_path / "simple.csv"
    path.write_text("x\nnot-an-integer\n", encoding="utf-8")

    with pytest.raises(ValueError, match="single character"):
        load_csv(path, delimiter="::")
    with pytest.raises(ValueError, match="missing_strategy"):
        load_csv(path, missing_strategy="ignore")
    with pytest.raises(ValueError, match="converted"):
        load_csv(path, dtype=int)


def test_iter_csv_reports_empty_file_and_missing_path(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="header row"):
        list(iter_csv(empty_path))
    with pytest.raises(OSError):
        list(iter_csv(tmp_path / "missing.csv"))
