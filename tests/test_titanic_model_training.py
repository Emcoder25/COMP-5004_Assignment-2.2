import numpy as np
import pytest

from demo.train_titanic import (
    DATA_PATH,
    FEATURE_COLUMNS,
    load_titanic_dataset,
    train_titanic,
)


def test_titanic_loader_reads_local_passengers_and_survival_labels():
    X, y = load_titanic_dataset()

    assert X.shape == (891, len(FEATURE_COLUMNS))
    assert y.shape == (891,)
    assert set(np.unique(y)) == {0, 1}
    assert "alive" not in FEATURE_COLUMNS
    assert DATA_PATH.is_file()


def test_titanic_model_trains_and_reports_held_out_metrics():
    result = train_titanic(test_size=0.25, random_state=11)

    assert result["train_samples"] == 668
    assert result["test_samples"] == 223
    assert result["train_samples"] + result["test_samples"] == 891
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["confusion_matrix"].shape == (2, 2)
    assert result["confusion_matrix"].sum() == result["test_samples"]
    assert result["model"].is_fitted_
    assert result["feature_columns"] == FEATURE_COLUMNS


def test_titanic_training_is_reproducible_for_fixed_random_state():
    first = train_titanic(test_size=0.2, random_state=123)
    second = train_titanic(test_size=0.2, random_state=123)

    assert first["accuracy"] == second["accuracy"]
    np.testing.assert_array_equal(
        first["confusion_matrix"], second["confusion_matrix"]
    )


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, np.nan, np.inf, True])
def test_titanic_training_rejects_invalid_test_size(test_size):
    with pytest.raises(ValueError, match="test_size"):
        train_titanic(test_size=test_size)


def test_titanic_loader_reports_missing_required_columns(tmp_path):
    path = tmp_path / "not_titanic.csv"
    path.write_text("survived,pclass\n0,3\n1,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_titanic_dataset(path)


def test_titanic_training_rejects_csv_with_too_few_rows(tmp_path):
    path = tmp_path / "one_passenger.csv"
    path.write_text(
        "survived,pclass,sex,age,sibsp,parch,fare,embarked\n"
        "1,1,female,30,0,0,20,S\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least two"):
        train_titanic(dataset_path=path)
