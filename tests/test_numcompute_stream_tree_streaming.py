import itertools
from pathlib import Path

import numpy as np
import pytest

from numcompute_stream.io import iter_csv
from numcompute_stream.metrics import accuracy
from numcompute_stream.pipeline import FeatureUnion, Pipeline
from numcompute_stream.preprocessing import ColumnSelector, Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTree, DecisionTreeClassifier


def test_tree_alias_class_order_predictions_and_probabilities():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["z", "z", "a", "a"])
    tree = DecisionTree(max_depth=2).fit(X, y)
    assert tree.classes_.tolist() == ["a", "z"]
    np.testing.assert_array_equal(tree.predict([[0.0], [3.0]]), ["z", "a"])
    probabilities = tree.predict_proba([[0.0], [3.0]])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities.shape == (2, 2)


def test_tree_buffering_state_and_prediction_before_fit():
    tree = DecisionTreeClassifier(buffer_size=4)
    with pytest.raises(ValueError, match="not fitted"):
        tree.predict([[0.0]])
    tree.partial_fit([[0.0], [1.0]], [0, 0])
    assert not tree.is_fitted_
    assert tree.buffered_sample_count_ == 2
    tree.partial_fit([[2.0], [3.0]], [1, 1])
    assert tree.is_fitted_
    assert tree.retraining_count_ == 1
    assert tree.buffered_sample_count_ == 0
    assert tree.n_leaves_ >= 1


def test_tree_keeps_previous_model_between_buffered_retrains():
    tree = DecisionTreeClassifier(buffer_size=2).fit([[0.0], [1.0]], [0, 1])
    original_retrains = tree.retraining_count_
    tree.partial_fit([[2.0]], [1])
    assert tree.retraining_count_ == original_retrains
    assert tree.predict([[0.0]]).shape == (1,)


def test_pipeline_uses_partial_fit_not_fit_for_incremental_estimator():
    class Model:
        def __init__(self):
            self.fit_calls = 0
            self.partial_calls = 0

        def fit(self, X, y):
            self.fit_calls += 1

        def partial_fit(self, X, y):
            self.partial_calls += 1
            self.label = int(np.asarray(y)[-1])

        def predict(self, X):
            return np.full(len(X), self.label)

    model = Model()
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    pipe.partial_fit([[0.0], [1.0]], [0, 1])
    pipe.partial_fit([[2.0]], [1])
    assert model.fit_calls == 0
    assert model.partial_calls == 2


def test_pipeline_feature_union_routes_mixed_columns_incrementally():
    X = np.array(
        [[1.0, "red"], [2.0, "blue"], [3.0, "green"], [4.0, "red"]],
        dtype=object,
    )
    union = FeatureUnion(
        [
            ("numeric", Pipeline([("select", ColumnSelector([0])), ("scale", StandardScaler())])),
            ("category", Pipeline([("select", ColumnSelector([1])), ("encode", OneHotEncoder())])),
        ]
    )
    # A transformer-only outer pipeline exercises branch-wise incremental updates.
    pipe = Pipeline([("features", union)])
    pipe.partial_fit(X[:2])
    first_width = pipe.transform(X[:2]).shape[1]
    pipe.partial_fit(X[2:])
    later_width = pipe.transform(X[2:]).shape[1]
    assert later_width == first_width + 1


def test_stream_trainer_prequential_scores_before_updating_each_chunk():
    class RecordingModel:
        def __init__(self):
            self.label = 0
            self.is_fitted_ = True
            self.events = []

        def predict(self, X):
            self.events.append(("predict", int(np.asarray(X)[0, 0])))
            return np.full(len(X), self.label)

        def partial_fit(self, X, y):
            self.events.append(("fit", int(np.asarray(X)[0, 0])))
            self.label = int(np.asarray(y)[0])

    model = RecordingModel()
    trainer = StreamTrainer(pipeline=model, track_memory=False)
    logs = trainer.run([(np.array([[1]]), np.array([1])), (np.array([[2]]), np.array([0]))])
    assert model.events == [("predict", 1), ("fit", 1), ("predict", 2), ("fit", 2)]
    assert logs[0]["accuracy"] == 0.0
    assert logs[1]["accuracy"] == 0.0
    np.testing.assert_allclose(trainer.metric_history("accuracy"), [0.0, 0.0])


def test_end_to_end_streaming_numeric_and_categorical_pipeline():
    X = np.array(
        [[0.0, "a"], [0.1, "b"], [0.8, "a"], [0.9, "c"], [0.2, "b"], [0.95, "c"]],
        dtype=object,
    )
    y = np.array([0, 0, 1, 1, 0, 1])
    features = FeatureUnion(
        [
            ("numeric", Pipeline([("select", ColumnSelector([0])), ("scale", StandardScaler())])),
            ("category", Pipeline([("select", ColumnSelector([1])), ("encode", OneHotEncoder(handle_unknown="ignore"))])),
        ]
    )
    model = DecisionTreeClassifier(max_depth=3, buffer_size=2)
    pipe = Pipeline([("features", features), ("tree", model)])
    trainer = StreamTrainer(pipe, track_memory=False)
    logs = trainer.run([(X[:2], y[:2]), (X[2:4], y[2:4]), (X[4:], y[4:])])
    assert len(logs) == 3
    assert model.is_fitted_
    assert accuracy(y, pipe.predict(X)) >= 0.8


def test_titanic_csv_runs_through_chunked_streaming_pipeline():
    data_path = Path(__file__).parents[1] / "demo" / "data" / "titanic.csv"
    chunks = iter_csv(data_path, chunksize=128)
    header, first_rows = next(chunks)
    feature_columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    feature_indices = [header.index(name) for name in feature_columns]
    target_index = header.index("survived")

    def stream():
        for rows in itertools.chain([first_rows], (chunk for _, chunk in chunks)):
            yield rows[:, feature_indices], rows[:, target_index].astype(int)

    features = FeatureUnion(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("select", ColumnSelector([0, 2, 3, 4, 5])),
                        ("impute", Imputer(strategy="mean")),
                        ("scale", StandardScaler()),
                    ]
                ),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("select", ColumnSelector([1, 6])),
                        ("impute", Imputer(strategy="constant", fill_value="Unknown")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
            ),
        ]
    )
    model = DecisionTreeClassifier(max_depth=4, buffer_size=128, random_state=7)
    pipeline = Pipeline([("features", features), ("tree", model)])
    trainer = StreamTrainer(pipeline, track_memory=False)
    logs = trainer.run(stream())

    assert "survived" not in feature_columns
    assert "alive" not in feature_columns
    assert sum(row["samples"] for row in logs) == 891
    assert logs[0]["accuracy"] is None
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in logs[1:])
    assert model.is_fitted_
    assert pipeline.predict(first_rows[:, feature_indices]).shape == (len(first_rows),)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_tree_criteria_fit_constant_class_data(criterion):
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=3).fit(
        [[0.0], [1.0], [2.0]], ["survived", "survived", "survived"]
    )

    assert tree.depth_ == 0
    assert tree.n_leaves_ == 1
    np.testing.assert_array_equal(tree.predict([[4.0], [5.0]]), ["survived", "survived"])


def test_tree_retrains_once_when_chunk_overshoots_buffer_threshold():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0.0], [1.0]], [0, 0])
    tree.partial_fit([[2.0], [3.0], [4.0]], [1, 1, 1])

    assert tree.retraining_count_ == 1
    assert tree.buffered_sample_count_ == 0


def test_tree_rejects_invalid_labels_and_nonfinite_features():
    with pytest.raises(ValueError, match="finite"):
        DecisionTreeClassifier().fit([[0.0], [1.0]], [0.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        DecisionTreeClassifier().fit([[0.0], [np.inf]], [0, 1])


def test_tree_depth_limit_and_empty_probability_output():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)

    assert tree.depth_ <= 1
    assert tree.predict_proba(np.empty((0, 1))).shape == (0, 2)


def test_pipeline_batch_fit_transform_predict_and_score():
    class MeanClassifier:
        def fit(self, X, y):
            self.label_ = int(np.mean(y) >= 0.5)
            return self

        def predict(self, X):
            return np.full(len(X), self.label_)

    pipe = Pipeline([("scale", StandardScaler()), ("model", MeanClassifier())])
    X = np.array([[1.0], [3.0], [5.0], [7.0]])
    y = np.array([0, 0, 1, 1])

    transformed = pipe.fit_transform(X, y)

    assert transformed.shape == X.shape
    assert pipe.score(X, y) == pytest.approx(0.5)


def test_pipeline_fit_resets_transformer_and_estimator_state():
    class CountingEstimator:
        def fit(self, X, y):
            self.fit_count = getattr(self, "fit_count", 0) + 1
            self.label_ = int(np.asarray(y)[0])
            return self

        def predict(self, X):
            return np.full(len(X), self.label_)

    scaler = StandardScaler()
    estimator = CountingEstimator()
    pipe = Pipeline([("scale", scaler), ("estimator", estimator)])
    pipe.fit([[1.0], [3.0]], [0, 0])
    pipe.fit([[10.0], [20.0]], [1, 1])

    assert scaler.n_samples_seen_ == 2
    assert scaler.mean_[0] == pytest.approx(15.0)
    assert estimator.fit_count == 2
    np.testing.assert_array_equal(pipe.predict([[10.0]]), [1])


def test_feature_union_concatenates_mixed_vector_and_matrix_outputs():
    class FirstColumn:
        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X)[:, 0]

    class LastColumn:
        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X)[:, 1:]

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    union = FeatureUnion([("first", FirstColumn()), ("rest", LastColumn())]).fit(X)

    np.testing.assert_array_equal(union.transform(X), X)


def test_feature_union_rejects_empty_component_list():
    with pytest.raises(ValueError, match="at least one"):
        FeatureUnion([])


def test_stream_trainer_cold_start_logs_no_fabricated_accuracy():
    tree = DecisionTreeClassifier(buffer_size=2)
    trainer = StreamTrainer(tree, track_memory=False)
    logs = trainer.run([(np.array([[0.0], [1.0]]), np.array([0, 1]))])

    assert logs[0]["accuracy"] is None
    assert logs[0]["samples"] == 2
    assert trainer.metric_history("accuracy").size == 0


def test_stream_trainer_logs_auc_from_model_probability_column():
    class ProbabilityModel:
        is_fitted_ = True
        classes_ = np.array(["lost", "survived"])

        def predict(self, X):
            return np.array(["lost", "survived"][: len(X)])

        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.2, 0.8]])[: len(X)]

        def partial_fit(self, X, y):
            return self

    from numcompute_stream.metrics import AUC

    trainer = StreamTrainer(
        ProbabilityModel(),
        additional_metrics={"auc": AUC(positive_label="survived")},
        track_memory=False,
    )
    score = trainer.score_chunk([[0.0], [1.0]], ["lost", "survived"])

    assert score == 1.0
    assert trainer.get_logs()[0]["auc"] == pytest.approx(1.0)


def test_stream_trainer_metric_history_omits_cold_start_and_is_copy():
    class Model:
        def __init__(self):
            self.is_fitted_ = False

        def partial_fit(self, X, y):
            self.is_fitted_ = True

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    trainer = StreamTrainer(Model(), track_memory=False)
    trainer.run(
        [
            (np.array([[0.0]]), np.array([0])),
            (np.array([[1.0]]), np.array([1])),
        ]
    )
    history = trainer.metric_history("accuracy")
    history[0] = -1

    np.testing.assert_array_equal(trainer.metric_history("accuracy"), [0.0])


def test_stream_trainer_score_rejects_prediction_shape_mismatch():
    class BadShapeModel:
        is_fitted_ = True

        def predict(self, X):
            return np.zeros((len(X), 1))

    with pytest.raises(ValueError, match="prediction shape"):
        StreamTrainer(BadShapeModel(), track_memory=False).score_chunk([[0.0]], [0])


def test_stream_trainer_rejects_conflicting_model_and_pipeline_arguments():
    model = object()

    with pytest.raises(ValueError, match="either model or pipeline"):
        StreamTrainer(model=model, pipeline=model)


def test_stream_trainer_returns_independent_log_dictionaries():
    class Model:
        is_fitted_ = True

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    trainer = StreamTrainer(Model(), track_memory=False)
    trainer.score_chunk([[0.0]], [0])
    copied_logs = trainer.get_logs()
    copied_logs[0]["accuracy"] = 99

    assert trainer.get_logs()[0]["accuracy"] == 1.0
