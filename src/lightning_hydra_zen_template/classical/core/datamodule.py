from abc import ABC, abstractmethod

from numpy.typing import ArrayLike


class DataModule(ABC):
    @abstractmethod
    def train_set(self) -> tuple[ArrayLike, ArrayLike]:
        """Return training data as (X, y) tuple of numpy arrays."""
        pass

    @abstractmethod
    def validation_set(self) -> tuple[ArrayLike, ArrayLike]:
        """Return validation data as (X, y) tuple of numpy arrays."""
        pass

    @abstractmethod
    def test_set(self) -> tuple[ArrayLike, ArrayLike]:
        """Return test data as (X, y) tuple of numpy arrays."""
        pass
