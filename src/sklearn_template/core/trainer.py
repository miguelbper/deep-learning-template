from pathlib import Path

from sklearn_template.core.datamodule import DataModule
from sklearn_template.core.model import Model


class Trainer:
    def __init__(self, ckpt_path: str | Path | None = None):
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None

    def fit(self, model: Model, datamodule: DataModule) -> None:
        X, y = datamodule.train_set()
        model.fit(X, y)
        if self.ckpt_path:
            model.save(self.ckpt_path)

    def validate(self, model: Model, datamodule: DataModule, ckpt_path: str | Path | None = None) -> None:
        X, y = datamodule.validation_set()
        if ckpt_path:
            model.load(ckpt_path)
        model.validate(X, y)

    def test(self, model: Model, datamodule: DataModule, ckpt_path: str | Path | None = None) -> None:
        X, y = datamodule.test_set()
        if ckpt_path:
            model.load(ckpt_path)
        model.test(X, y)
