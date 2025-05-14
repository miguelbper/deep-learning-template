from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from lightning_hydra_zen_template.classical.core.datamodule import DataModule
from lightning_hydra_zen_template.classical.core.model import Estimator, Metric, Model
from lightning_hydra_zen_template.classical.core.trainer import Trainer

NUM_TRAIN_SAMPLES = 10
NUM_VAL_SAMPLES = 5
NUM_TEST_SAMPLES = 5
NUM_FEATURES = 5
NUM_CLASSES = 2


class TestDataModule(DataModule):
    def __init__(self):
        self._X_train = np.random.rand(NUM_TRAIN_SAMPLES, NUM_FEATURES)
        self._y_train = np.random.randint(0, NUM_CLASSES, NUM_TRAIN_SAMPLES)

        self._X_val = np.random.rand(NUM_VAL_SAMPLES, NUM_FEATURES)
        self._y_val = np.random.randint(0, NUM_CLASSES, NUM_VAL_SAMPLES)

        self._X_test = np.random.rand(NUM_TEST_SAMPLES, NUM_FEATURES)
        self._y_test = np.random.randint(0, NUM_CLASSES, NUM_TEST_SAMPLES)

    def train_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X_train, self._y_train

    def validation_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X_val, self._y_val

    def test_set(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X_test, self._y_test


@pytest.fixture
def linreg() -> Estimator:
    return LinearRegression()


@pytest.fixture
def mse() -> Metric:
    return mean_squared_error


@pytest.fixture
def model(linreg: Estimator, mse: Metric) -> Model:
    return Model(linreg, [mse])


@pytest.fixture
def datamodule() -> DataModule:
    return TestDataModule()


@pytest.fixture
def ckpt_path(tmp_path: Path) -> Path:
    return tmp_path / "model.pkl"


@pytest.fixture
def trainer(ckpt_path: Path) -> Trainer:
    return Trainer(ckpt_path=ckpt_path)


class TestTrainer:
    def test_init(self, ckpt_path: Path) -> None:
        trainer = Trainer(ckpt_path=ckpt_path)
        assert trainer.ckpt_path == ckpt_path

        trainer_no_ckpt = Trainer()
        assert trainer_no_ckpt.ckpt_path is None

    @patch("builtins.print")
    def test_print_metrics(self, mock_print, trainer: Trainer) -> None:
        metrics = {"metric1": 0.95, "metric2": 0.85}
        trainer.print_metrics(metrics, prefix="Test")
        assert mock_print.call_count >= 3

    def test_fit(self, trainer: Trainer, model: Model, datamodule: DataModule, ckpt_path: Path) -> None:
        trainer.fit(model, datamodule)
        assert hasattr(model.model, "coef_")
        assert hasattr(model.model, "intercept_")
        assert ckpt_path.exists()

    def test_validate(self, trainer: Trainer, model: Model, datamodule: DataModule) -> None:
        trainer.fit(model, datamodule)
        metrics = trainer.validate(model, datamodule)
        assert isinstance(metrics, dict)
        assert len(metrics) == 1
        assert "val/mean_squared_error" in metrics
        assert isinstance(metrics["val/mean_squared_error"], float)

    def test_test(self, trainer: Trainer, model: Model, datamodule: DataModule) -> None:
        trainer.fit(model, datamodule)
        metrics = trainer.test(model, datamodule)
        assert isinstance(metrics, dict)
        assert len(metrics) == 1
        assert "test/mean_squared_error" in metrics
        assert isinstance(metrics["test/mean_squared_error"], float)

    def test_load_from_checkpoint(self, tmp_path: Path, model: Model, datamodule: DataModule) -> None:
        ckpt_path = tmp_path / "model.pkl"
        trainer = Trainer(ckpt_path=ckpt_path)
        trainer.fit(model, datamodule)

        val_metrics_ckpt = trainer.validate(None, datamodule, ckpt_path=ckpt_path)
        val_metrics_model = trainer.validate(model, datamodule, ckpt_path=None)
        assert val_metrics_ckpt.keys() == val_metrics_model.keys()
        assert val_metrics_ckpt == val_metrics_model

        test_metrics_ckpt = trainer.test(None, datamodule, ckpt_path=ckpt_path)
        test_metrics_model = trainer.test(model, datamodule, ckpt_path=None)
        assert test_metrics_ckpt.keys() == test_metrics_model.keys()
        assert test_metrics_ckpt == test_metrics_model
