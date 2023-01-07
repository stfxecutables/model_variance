from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import Literal

from src.enumerables import RuntimeClass
from src.hparams.hparams import Hparams
from src.models.model import ClassifierModel


def loader(
    X: ndarray, y: ndarray, batch_size: int = 1024, shuffle: bool = False
) -> DataLoader:

    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=1024, shuffle=shuffle)


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
    def __init__(self, hparams: Hparams, logdir: Path, runtime: RuntimeClass) -> None:
        super().__init__(hparams=hparams, runtime=runtime, logdir=logdir)
        self.model: LightningModule
        self.trainer: Trainer | None = None

    def fit(self, X: ndarray, y: ndarray) -> None:
        train_loader = loader(X, y, shuffle=True)
        accel = "gpu" if torch.cuda.is_available() else "cpu"
        self.trainer = Trainer(
            logger=False,
            callbacks=callbacks(),
            accelerator=accel,
            enable_checkpointing=False,
            max_epochs=5,
        )
        self.trainer.fit(self.model, train_dataloaders=train_loader)
        self.fitted = True

    def predict(self, X: ndarray, y: ndarray) -> ndarray:
        if not self.fitted:
            raise RuntimeError("Model has not yet been fitted.")

        test_loader = loader(X=X, y=y, shuffle=False)
        trainer = self.trainer
        results: list[dict[str, ndarray]] = trainer.test(
            model=self.model, dataloaders=test_loader, ckpt_path="last"
        )
        preds = np.concatenate([result["pred"] for result in results], axis=0)
        # targs = np.concatenate([result["target"] for result in results], axis=0)
        return preds
