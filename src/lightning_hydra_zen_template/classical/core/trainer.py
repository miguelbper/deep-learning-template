from pathlib import Path

from lightning_hydra_zen_template.classical.core.datamodule import DataModule
from lightning_hydra_zen_template.classical.core.model import Model
from lightning_hydra_zen_template.classical.utils.print_metrics import print_metrics

Ckpt = str | Path
Metrics = dict[str, float]


class Trainer:
    def fit(self, model: Model, datamodule: DataModule, ckpt_path: Ckpt | None = None) -> None:
        X, y = datamodule.train_set()
        model.train(X, y)
        if ckpt_path:
            model.save(ckpt_path)

    def validate(self, model: Model | None, datamodule: DataModule, ckpt_path: Ckpt | None = None) -> Metrics:
        if model is None and ckpt_path is None:
            raise ValueError("Either model or ckpt_path must be provided")
        if ckpt_path:
            model = Model.load(ckpt_path)
        if not model.trained:
            raise ValueError("Model must be trained before validation")
        X, y = datamodule.validation_set()
        metrics = model.validate(X, y)
        print_metrics(metrics, "Validation")
        return metrics

    def test(self, model: Model | None, datamodule: DataModule, ckpt_path: Ckpt | None = None) -> Metrics:
        if model is None and ckpt_path is None:
            raise ValueError("Either model or ckpt_path must be provided")
        if ckpt_path:
            model = Model.load(ckpt_path)
        if not model.trained:
            raise ValueError("Model must be trained before testing")
        X, y = datamodule.test_set()
        metrics = model.test(X, y)
        print_metrics(metrics, "Test")
        return metrics
