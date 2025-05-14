from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import sklearn

Estimator = sklearn.base.BaseEstimator
Metric = Callable[[np.ndarray, np.ndarray], float]


class Model:
    def __init__(self, model: Estimator, metrics: list[Metric]):
        self.model = model
        self.metrics = metrics

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data X."""
        return self.model.predict(X)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on input features X and target y."""
        self.model.fit(X, y)

    def evaluate(self, X: np.ndarray, y: np.ndarray, prefix: str) -> dict[str, float]:
        """Evaluate the model on input features X and target y.

        Args:
            X: Input features
            y: Target values
            prefix: Prefix for metric names (e.g., 'val_' or 'test_')

        Returns:
            dict[str, float]: Dictionary with metric names (prefixed) as keys and their values.
        """
        y_pred = self(X)
        results = {}
        for metric in self.metrics:
            metric_name = f"{prefix}{metric.__name__}"
            metric_value = metric(y, y_pred)
            results[metric_name] = metric_value
        return results

    def validate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Validate the model on input features X and target y.

        Returns:
            dict[str, float]: Dictionary with metric names (prefixed with 'val/') as keys and their values.
        """
        return self.evaluate(X, y, prefix="val/")

    def test(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Test the model on input features X and target y.

        Returns:
            dict[str, float]: Dictionary with metric names (prefixed with 'test/') as keys and their values.
        """
        return self.evaluate(X, y, prefix="test/")

    def save(self, path: str | Path) -> None:
        """Save the entire Model object to the specified path.

        Args:
            path: Path where to save the model.
        """
        path = Path(path)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Model":
        """Load a Model object from the specified path.

        Args:
            path: Path from where to load the model.

        Returns:
            Model: The loaded model object.
        """
        path = Path(path)
        return joblib.load(path)
