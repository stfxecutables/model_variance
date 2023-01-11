from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path
from typing import Callable, Mapping
from warnings import catch_warnings, filterwarnings

import numpy as np
import torch
from numpy import ndarray
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader, TensorDataset

from src.constants import BATCH_SIZE
from src.dataset import Dataset
from src.hparams.hparams import Hparams
from src.models.model import ClassifierModel


class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return


def loader(
    X: ndarray, y: ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = False
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X).clone().detach().float().contiguous(),
        torch.from_numpy(y).clone().detach().long().contiguous(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def callbacks() -> list[Callable]:
    return [
        ModelCheckpoint(
            monitor="train/acc", save_last=True, mode="max", every_n_epochs=1
        ),
        EarlyStopping(
            monitor="train/acc", patience=999999, mode="max", check_finite=True
        ),
    ]


class DLModel(ClassifierModel):
    def __init__(self, hparams: Hparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__(hparams=hparams, dataset=dataset, logdir=logdir)
        self.model: LightningModule
        self.trainer: Trainer | None = None
        self.max_epochs: int
        self.fitted: bool = False

    def fit(self, X: ndarray, y: ndarray) -> None:
        train_loader = loader(X, y, shuffle=True)
        accel = "gpu" if torch.cuda.is_available() else "cpu"
        self.trainer = Trainer(
            logger=TensorBoardLogger(str(self.logdir), name=None),
            callbacks=callbacks(),
            accelerator=accel,
            enable_checkpointing=True,
            max_epochs=self.max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
        )
        with catch_warnings():
            # stfu Lightning
            filterwarnings("ignore", message="You defined a `validation_step`")
            filterwarnings("ignore", message="The dataloader")
            filterwarnings("ignore", message="The number of training batches")
            self.model = self.model_cls(**self._get_model_args())
            self.trainer.fit(self.model, train_dataloaders=train_loader)
        self.fitted = True

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        if self.fitted is False:
            raise RuntimeError("Model has not yet been fitted.")

        test_loader = loader(X=X, y=y, shuffle=False)
        trainer = self.trainer
        with catch_warnings():
            # stfu Lightning
            filterwarnings("ignore", message="The dataloader")
            results: list[dict[str, ndarray]] = trainer.predict(
                model=self.model,
                dataloaders=test_loader,
                ckpt_path="last",
                return_predictions=True,
            )
        preds = np.concatenate([result["pred"] for result in results], axis=0)
        targs = np.concatenate([result["target"] for result in results], axis=0)
        return preds, targs

    def _get_model_args(self) -> Mapping:
        return dict(
            dataset=self.dataset,
            log_version_dir=self.logdir,
            hps=self.hparams,
        )
