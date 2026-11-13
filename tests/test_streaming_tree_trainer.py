import numpy as np
import pytest

from numcompute.stream import StreamTrainer
from numcompute.tree import DecisionTreeClassifier


def test_tree_fit_learns_separable_numeric_classes():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)

    np.testing.assert_array_equal(tree.predict([[0.2], [2.8]]), [0, 1])
    assert tree.score(X, y) == pytest.approx(1.0)


def test_tree_partial_fit_waits_for_buffer_before_initial_training():
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[0.0], [1.0]], [0, 0])

    with pytest.raises(ValueError, match="not been fitted"):
        tree.predict([[0.0]])

    tree.partial_fit([[2.0], [3.0]], [1, 1])
    assert tree.root_ is not None
    np.testing.assert_array_equal(tree.predict([[0.0], [3.0]]), [0, 1])


def test_tree_retrains_at_buffer_threshold_and_clears_buffer():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0.0], [1.0]], [0, 0])
    assert sum(chunk.shape[0] for chunk in tree._buffer_X) == 2

    tree.partial_fit([[2.0]], [1])

    assert tree._buffer_X == []
    assert tree._buffer_y == []
    assert tree.root_ is not None


def test_tree_retains_last_model_between_retraining_thresholds():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0.0], [1.0]], [0, 1])
    previous_root = tree.root_

    tree.partial_fit([[2.0]], [1])

    assert tree.root_ is previous_root
    assert tree._buffer_X[0].shape[0] == 1


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        ([[1.0]], [[0]], "y must be a 1D array"),
        ([[1.0], [2.0]], [0], "same number of samples"),
        ([1.0, 2.0], [0, 1], "2D array"),
    ],
)
def test_tree_fit_reports_invalid_shapes(X, y, message):
    with pytest.raises(ValueError, match=message):
        DecisionTreeClassifier().fit(X, y)


def test_tree_rejects_invalid_buffer_and_criterion():
    with pytest.raises(ValueError, match="buffer_size"):
        DecisionTreeClassifier(buffer_size=0)
    with pytest.raises(ValueError, match="criterion"):
        DecisionTreeClassifier(criterion="unsupported")


def test_stream_trainer_fits_and_scores_chunks():
    class IncrementalClassifier:
        def partial_fit(self, X, y):
            self.label_ = int(np.asarray(y)[0])

        def predict(self, X):
            return np.full(len(X), self.label_)

    trainer = StreamTrainer(model=IncrementalClassifier())
    trainer.fit_chunk([[0.0], [1.0]], [1, 1])
    assert trainer.score_chunk([[2.0], [3.0]], [1, 0]) == pytest.approx(0.5)
    assert trainer.get_logs()[0]["chunk"] == 0


def test_stream_trainer_errors_when_no_model_or_predictor():
    with pytest.raises(ValueError, match="model or pipeline"):
        StreamTrainer().fit_chunk([[0.0]], [0])

    trainer = StreamTrainer(model=object())
    with pytest.raises(AttributeError, match="predict"):
        trainer.score_chunk([[0.0]], [0])
