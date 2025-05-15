from hydra_zen import make_config, make_custom_builds_fn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from lightning_hydra_zen_template.classical.core.module import Model
from lightning_hydra_zen_template.classical.core.trainer import Trainer
from lightning_hydra_zen_template.classical.data.iris import IrisDataModule

fbuilds = make_custom_builds_fn(populate_full_signature=True)

TrainCfg = make_config(
    data=fbuilds(IrisDataModule),
    model=fbuilds(
        Model,
        model=fbuilds(LogisticRegression),
        metrics=[
            fbuilds(accuracy_score),
        ],
    ),
    trainer=fbuilds(Trainer),
    monitor="val/accuracy_score",
)
