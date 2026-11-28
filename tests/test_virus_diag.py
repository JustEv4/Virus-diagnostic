import pandas as pd
import pytest

from core.dataset import DatasetParser
from pipeline.Trainer import Trainer
from pipeline.ClinicalPredictor import ClinicalPredictor

DATA = "data/clinical_infection_dataset.csv"


def test_dataset_parse():
    X, y = DatasetParser(DATA).parse()
    assert "infection" not in X.columns
    assert len(X) == len(y) > 0


def test_missing_target_column(tmp_path):
    f = tmp_path / "bad.csv"
    pd.DataFrame({"age": [1, 2]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        DatasetParser(str(f)).load()


def test_trainer_returns_model_and_test_set():
    model, X_test, y_test = Trainer(DATA).train()
    assert len(X_test) == len(y_test) > 0
    proba = model.predict_proba(X_test.iloc[:1])
    assert 0.0 <= proba[0][1] <= 1.0


@pytest.mark.parametrize("threshold, expected", [(0.0, "infecté"), (1.01, "sain")])
def test_predictor_threshold(threshold, expected):
    model, X_test, _ = Trainer(DATA).train()
    predictor = ClinicalPredictor(model, threshold)
    assert predictor.diagnose(X_test.iloc[:1]) == expected