from abc import ABC, abstractmethod

import numpy as np


class DataModule(ABC):
    @abstractmethod
    def train_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Return training data as (X, y) tuple of numpy arrays."""
        pass

    @abstractmethod
    def validation_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Return validation data as (X, y) tuple of numpy arrays."""
        pass

    @abstractmethod
    def test_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Return test data as (X, y) tuple of numpy arrays."""
        pass
