from pathlib import Path

from lightning_hydra_zen_template.classical.core.datamodule import DataModule
from lightning_hydra_zen_template.classical.core.module import Module
from lightning_hydra_zen_template.classical.utils.print_metrics import print_metrics

Ckpt = str | Path
Metrics = dict[str, float]


class Trainer:
    def fit(self, model: Module, datamodule: DataModule, ckpt_path: Ckpt | None = None) -> None:
        X, y = datamodule.train_set()
        model.train(X, y)
        if ckpt_path:
            model.save(ckpt_path)

    def evaluate(self, model: Module | None, datamodule: DataModule, ckpt_path: Ckpt | None, split: str) -> Metrics:
        """Common evaluation logic for validation and test.

        Args:
            model: Model to evaluate
            datamodule: DataModule containing the data
            ckpt_path: Optional path to load model from
            split: Which split to evaluate ('validation' or 'test')

        Returns:
            Metrics from the evaluation
        """
        if model is None and ckpt_path is None:
            raise ValueError("Either model or ckpt_path must be provided")
        if ckpt_path:
            model = Module.load(ckpt_path)
        if not model.trained:
            raise ValueError("Model must be trained before evaluation")

        if split == "validation":
            X, y = datamodule.validation_set()
            metrics = model.validate(X, y)
            print_metrics(metrics, "Validation")
        else:  # test
            X, y = datamodule.test_set()
            metrics = model.test(X, y)
            print_metrics(metrics, "Test")

        return metrics

    def validate(self, model: Module | None, datamodule: DataModule, ckpt_path: Ckpt | None = None) -> Metrics:
        return self.evaluate(model, datamodule, ckpt_path, "validation")

    def test(self, model: Module | None, datamodule: DataModule, ckpt_path: Ckpt | None = None) -> Metrics:
        return self.evaluate(model, datamodule, ckpt_path, "test")
