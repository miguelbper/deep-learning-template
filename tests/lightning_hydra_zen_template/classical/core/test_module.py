from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from lightning_hydra_zen_template.classical.core.module import Estimator, Metric, Model

N = 10
NUM_SAMPLES = N - 1
NUM_FEATURES = N


@pytest.fixture
def X() -> np.ndarray:
    return np.random.rand(NUM_SAMPLES, NUM_FEATURES)


@pytest.fixture
def y() -> np.ndarray:
    return np.random.rand(NUM_SAMPLES)


@pytest.fixture
def linreg(X: np.ndarray, y: np.ndarray) -> Estimator:
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    return lin_reg


@pytest.fixture
def mse() -> Metric:
    return mean_squared_error


@pytest.fixture
def model(linreg: Estimator, mse: Metric) -> Model:
    return Model(linreg, [mse])


class TestModel:
    def test_model_init(self, linreg: Estimator, mse: Metric) -> None:
        model = Model(linreg, [mse])
        assert isinstance(model, Model)

    def test_model_call(self, model: Model, X: np.ndarray, y: np.ndarray) -> None:
        y_pred_0 = model(X)
        y_pred_1 = model.model.predict(X)
        assert np.allclose(y_pred_0, y_pred_1)

    def test_model_train(self, model: Model, X: np.ndarray, y: np.ndarray) -> None:
        weights_0 = model.model.coef_
        bias_0 = model.model.intercept_

        model.train(X, y)
        weights_1 = model.model.coef_
        bias_1 = model.model.intercept_

        assert np.allclose(weights_0, weights_1)
        assert np.allclose(bias_0, bias_1)
        assert isinstance(weights_0, np.ndarray)
        assert isinstance(bias_0, float)

    def test_model_evaluate(self, model: Model, X: np.ndarray, y: np.ndarray) -> None:
        results = model.evaluate(X, y, prefix="custom_")
        assert isinstance(results, dict)
        assert len(results) == 1
        assert "custom_mean_squared_error" in results
        assert np.isclose(results["custom_mean_squared_error"], 0.0)

    def test_model_validate(self, model: Model, X: np.ndarray, y: np.ndarray) -> None:
        val_results = model.validate(X, y)
        assert isinstance(val_results, dict)
        assert len(val_results) == 1
        assert "val/mean_squared_error" in val_results
        assert np.isclose(val_results["val/mean_squared_error"], 0.0)

    def test_model_test(self, model: Model, X: np.ndarray, y: np.ndarray) -> None:
        test_results = model.test(X, y)
        assert isinstance(test_results, dict)
        assert len(test_results) == 1
        assert "test/mean_squared_error" in test_results
        assert np.isclose(test_results["test/mean_squared_error"], 0.0)

    def test_model_save(self, model: Model, tmp_path: Path) -> None:
        save_path = tmp_path / "model.pkl"
        model.save(save_path)
        assert save_path.exists()

    def test_model_load(self, model: Model, X: np.ndarray, tmp_path: Path) -> None:
        save_path = tmp_path / "model.pkl"
        model.save(save_path)
        loaded_model = Model.load(save_path)
        assert isinstance(loaded_model, Model)
        assert np.allclose(model.model.coef_, loaded_model.model.coef_)
        assert np.allclose(model.model.intercept_, loaded_model.model.intercept_)
        assert np.allclose(model(X), loaded_model(X))
        assert len(model.metrics) == len(loaded_model.metrics)
        assert model.metrics[0].__name__ == loaded_model.metrics[0].__name__
