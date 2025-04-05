from pathlib import Path

import rootutils
from hydra.conf import HydraConf, RunDir, SweepDir
from hydra_zen import builds, make_config, make_custom_builds_fn
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
)

from src.datamodule.mnist import MNISTDataModule
from src.model.model import Model
from src.model.resnet import ResNet

pbuilds = make_custom_builds_fn(zen_partial=True)
root_dir = rootutils.find_root(search_from=__file__)


DataModuleCfg = builds(
    MNISTDataModule,
    data_dir="${paths.data_dir}",
)


ModelCfg = builds(
    Model,
    model=builds(
        ResNet,
        num_classes=10,
    ),
    loss_fn=builds(
        nn.CrossEntropyLoss,
    ),
    optimizer=pbuilds(
        Adam,
    ),
    scheduler=pbuilds(
        ReduceLROnPlateau,
        mode="min",
        factor=0.1,
        patience=10,
    ),
    metric_collection=builds(
        MetricCollection,
        metrics=[
            builds(
                Accuracy,
                task="multiclass",
                num_classes=10,
                average="micro",
            ),
            builds(
                F1Score,
                task="multiclass",
                num_classes=10,
                average="macro",
            ),
            builds(
                Precision,
                task="multiclass",
                num_classes=10,
                average="macro",
            ),
            builds(
                Recall,
                task="multiclass",
                num_classes=10,
                average="macro",
            ),
        ],
    ),
)


TrainerCfg = builds(
    Trainer,
    logger=[
        builds(
            CSVLogger,
            save_dir="${paths.output_dir}",
            name="csv",
        ),
        builds(
            MLFlowLogger,
            tracking_uri=str(Path("${paths.log_dir}") / "mlflow" / "mlruns"),
        ),
    ],
    callbacks=[
        builds(
            EarlyStopping,
            monitor="${monitor}",
            patience=3,
            mode="${mode}",
        ),
        builds(
            ModelCheckpoint,
            dirpath=str(Path("${paths.output_dir}") / "checkpoints"),
            monitor="${monitor}",
            save_last=True,
            mode="${mode}",
            auto_insert_metric_name=False,
        ),
        builds(RichProgressBar),
        builds(
            RichModelSummary,
            max_depth=-1,
        ),
    ],
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir="${paths.output_dir}",
)


HydraCfg = HydraConf(
    run=RunDir(str(Path("${paths.log_dir}") / "hydra" / "runs" / "${now:%Y-%m-%d}" / "${now:%H-%M-%S}")),
    sweep=SweepDir(
        dir=str(Path("${paths.log_dir}") / "hydra" / "multiruns" / "${now:%Y-%m-%d}" / "${now:%H-%M-%S}"),
        subdir="${hydra.job.num}",
    ),
)


Paths = make_config(
    root_dir=str(root_dir),
    data_dir=str(root_dir / "data" / "processed"),
    log_dir=str(root_dir / "logs"),
    output_dir="${hydra:runtime.output_dir}",
    work_dir="${hydra:runtime.cwd}",
)


RunConfig = make_config(
    evaluate_after_train=True,
    ckpt_path=None,
    seed=42,
    monitor="val/MulticlassAccuracy",
    mode="max",
)


Config = make_config(
    datamodule=DataModuleCfg,
    model=ModelCfg,
    trainer=TrainerCfg,
    hydra=HydraCfg,
    paths=Paths,
    bases=(RunConfig,),
)
