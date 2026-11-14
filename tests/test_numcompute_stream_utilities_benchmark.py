import numpy as np
import pytest

from numcompute_stream.benchmarking import benchmark, compare_implementations
from numcompute_stream.optim import grad, jacobian
from numcompute_stream.rank import StreamingRanker, StreamingTopK, percentile, rank
from numcompute_stream.utils import euclidean_distance, manhattan_distance, positive_int


def test_positive_integer_validation_rejects_booleans():
    assert positive_int(np.int64(2), "n") == 2
    with pytest.raises(TypeError, match="positive integer"):
        positive_int(True, "n")


def test_distance_helpers_handle_large_finite_values_and_reject_nonfinite():
    assert np.isfinite(euclidean_distance([1e150, 0.0], [0.0, 1e150]))
    with pytest.raises(ValueError, match="finite"):
        manhattan_distance([np.inf], [0.0])


def test_gradient_uses_representable_step_at_large_magnitude():
    derivative = grad(lambda value: value / 1e300, 1e300, h=1e-10)
    assert derivative == pytest.approx(1e-300, rel=1e-8)


def test_gradient_and_jacobian_reject_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        grad(lambda x: np.inf, 1.0)
    with pytest.raises(ValueError, match="finite"):
        grad(lambda x: x, np.nan)
    np.testing.assert_allclose(jacobian(lambda x: [x[0] ** 2, x[0]], [3.0]), [[6.0], [1.0]], rtol=1e-5)


def test_streaming_topk_stable_ties_and_order_for_each_direction():
    largest = StreamingTopK(3).update([4, 5]).update([5, 3])
    values, indices = largest.result()
    np.testing.assert_array_equal(values, [5, 5, 4])
    np.testing.assert_array_equal(indices, [1, 2, 0])
    smallest = StreamingTopK(2, largest=False).update([3, 1, 1])
    np.testing.assert_array_equal(smallest.result()[1], [1, 2])


def test_streaming_ranker_handles_ties_across_chunks():
    ranker = StreamingRanker("average").update([2, 1]).update([2, 3])
    np.testing.assert_allclose(ranker.result(), [2.5, 1.0, 2.5, 4.0])


def test_benchmark_consumes_lazy_iterator_and_compares_outputs():
    result = benchmark(lambda: iter([1, 2, 3]), repeats=2, warmup=0)
    assert result["lazy_iterators_consumed"] is True
    compared = compare_implementations(
        {"reference": lambda x: x + 1, "same": lambda x: x + 1},
        np.array([1, 2]),
        reference="reference",
        repeats=1,
        warmup=0,
    )
    assert set(compared) == {"reference", "same"}


def test_benchmark_rejects_bad_timing_parameters_and_wrong_outputs():
    with pytest.raises(ValueError, match="repeats"):
        benchmark(lambda: None, repeats=0)
    with pytest.raises(ValueError, match="does not match"):
        compare_implementations(
            {"reference": lambda: np.array([0]), "bad": lambda: np.array([1])},
            reference="reference",
            repeats=1,
            warmup=0,
        )


def test_streaming_topk_returns_all_available_values_when_k_exceeds_stream():
    values, indices = StreamingTopK(5).update([2, 8, 4]).result()

    np.testing.assert_array_equal(values, [8, 4, 2])
    np.testing.assert_array_equal(indices, [1, 2, 0])


def test_streaming_topk_rejects_invalid_shapes_and_nonfinite_values():
    with pytest.raises(TypeError, match="1D"):
        StreamingTopK(2).update([[1, 2]])
    with pytest.raises(ValueError, match="finite"):
        StreamingTopK(2).update([1.0, np.inf])


def test_streaming_ranker_rejects_empty_result_and_nonfinite_input():
    with pytest.raises(ValueError, match="No values"):
        StreamingRanker().result()
    with pytest.raises(ValueError, match="finite"):
        StreamingRanker().update([1.0, np.nan])


def test_batch_rank_and_percentile_match_expected_ties():
    np.testing.assert_allclose(rank([10, 20, 20, 40]), [1.0, 2.5, 2.5, 4.0])
    assert percentile([0, 10, 20, 30], 50) == pytest.approx(15.0)


def test_gradient_forward_method_and_invalid_step_are_checked():
    assert grad(lambda x: x * x, 3.0, method="forward") == pytest.approx(6.0, rel=1e-4)
    with pytest.raises(ValueError, match="positive and finite"):
        grad(lambda x: x, 1.0, h=0.0)


def test_jacobian_returns_expected_shape_for_vector_output():
    result = jacobian(lambda x: [x[0] + x[1], x[0] * x[1]], [2.0, 3.0])

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, [[1.0, 1.0], [3.0, 2.0]], rtol=1e-5)


def test_benchmark_warmups_are_excluded_from_reported_repeat_count():
    calls = []

    result = benchmark(lambda: calls.append(len(calls)), repeats=3, warmup=2)

    assert len(calls) == 5
    assert result["repeats"] == 3
    assert result["minimum_seconds"] <= result["median_seconds"]
