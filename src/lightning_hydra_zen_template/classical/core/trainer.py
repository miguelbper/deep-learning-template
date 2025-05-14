from pathlib import Path

from tabulate import tabulate

from lightning_hydra_zen_template.classical.core.datamodule import DataModule
from lightning_hydra_zen_template.classical.core.model import Model


class Trainer:
    def __init__(self, ckpt_path: str | Path | None = None):
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None

    def print_metrics(self, metrics: dict[str, float], prefix: str) -> None:
        """Pretty print metrics in a table format.

        Args:
            metrics: Dictionary of metric names and values
            prefix: Prefix to use in the title (e.g., 'Validation' or 'Test')
        """
        table = [[name, f"{value:.4f}"] for name, value in metrics.items()]
        print(f"\n{prefix} Metrics:")
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
        print()

    def fit(self, model: Model, datamodule: DataModule) -> None:
        X, y = datamodule.train_set()
        model.fit(X, y)
        if self.ckpt_path:
            model.save(self.ckpt_path)

    def validate(self, model: Model, datamodule: DataModule, ckpt_path: str | Path | None = None) -> dict[str, float]:
        X, y = datamodule.validation_set()
        if ckpt_path:
            model.load(ckpt_path)
        metrics = model.validate(X, y)
        self.print_metrics(metrics, "Validation")
        return metrics

    def test(self, model: Model, datamodule: DataModule, ckpt_path: str | Path | None = None) -> dict[str, float]:
        X, y = datamodule.test_set()
        if ckpt_path:
            model.load(ckpt_path)
        metrics = model.test(X, y)
        self.print_metrics(metrics, "Test")
        return metrics
