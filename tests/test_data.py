import pandas as pd
import pytest

from mlops_project.data import descriptive_summary, load_dataset


def test_descriptive_summary_contains_columns():
    summary = descriptive_summary(pd.DataFrame({"age": [20, 30], "label": [0, 1]}))
    assert set(summary.index) == {"age", "label"}


def test_load_dataset_requires_target(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"feature": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cible"):
        load_dataset(path, "target")
