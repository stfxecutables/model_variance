from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import logging
import sys
from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from platform import system
from typing import Dict, List, Mapping, Optional
from warnings import catch_warnings, filterwarnings

import numpy as np
import torch
from numpy import ndarray
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment  # type: ignore
from torch.utils.data import DataLoader, TensorDataset

from src.constants import BATCH_SIZE
from src.dataset import Dataset
from src.enumerables import RuntimeClass
from src.hparams.hparams import Hparams
from src.models.model import ClassifierModel


class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect(self) -> bool:
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
        torch.from_numpy(y).clone().detach().to(torch.int32).contiguous(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def callbacks() -> List[Callback]:
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
        self.model: Optional[LightningModule] = None
        self.trainer: Optional[Trainer] = None
        self.max_epochs: int
        self.fitted: bool = False
        self.fitted_model: Path = self.logdir / "fitted.pickle"
        self.loaded = False

    def load_fitted(self) -> None:
        if (self.trainer is not None) or self.fitted:
            raise RuntimeError("DLModel already appears to have a fitted model.")
        self.trainer = self._setup_trainer(pretrained=True)
        self.model = torch.load(self.fitted_model, map_location="cpu")
        self.fitted = True
        self.loaded = True

    def batch_size(self) -> int:
        is_fast = self.dataset.name in RuntimeClass.Fast.members()
        is_mid = self.dataset.name in RuntimeClass.Mid.members()
        is_slow = self.dataset.name in RuntimeClass.Slow.members()
        batch_size = BATCH_SIZE
        if is_slow:
            batch_size = 2048
        elif is_mid:
            batch_size = 1024
        elif is_fast:
            batch_size = 64
        return batch_size

    def fit(self, X: ndarray, y: ndarray, save: bool = True) -> None:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        batch_size = self.batch_size()
        if self.trainer is not None:
            raise RuntimeError("Cannot fit an already-fit model.")
        self.trainer = self._setup_trainer(pretrained=False)

        train_loader = loader(X, y, batch_size=batch_size, shuffle=True)
        with catch_warnings():
            # stfu Lightning
            filterwarnings("ignore", message="You defined a `validation_step`")
            filterwarnings("ignore", message="The dataloader")
            filterwarnings("ignore", message="The number of training batches")
            filterwarnings("ignore", message="MPS: no support for int64")
            self.model = self.model_cls(**self._get_model_args())
            self.trainer.fit(self.model, train_dataloaders=train_loader)
        # self.model is either a Linear or Sequential, should be no issue
        if save:
            torch.save(self.model, f=self.fitted_model, pickle_protocol=HIGHEST_PROTOCOL)
        self.fitted = True

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        if self.fitted is False or self.trainer is None or self.model is None:
            raise RuntimeError("Model has not yet been fitted.")

        test_loader = loader(X=X, y=y, shuffle=False)
        trainer: Trainer = self.trainer
        with catch_warnings():
            # stfu Lightning
            filterwarnings("ignore", message="The dataloader")
            results: List[Dict[str, ndarray]] = trainer.predict(  # type: ignore
                model=self.model,
                dataloaders=test_loader,
                ckpt_path="last" if not self.loaded else None,
                return_predictions=True,
            )
        preds = np.concatenate([result["pred"] for result in results], axis=0)
        targs = np.concatenate([result["target"] for result in results], axis=0)
        return preds, targs

    def _setup_trainer(self, pretrained: bool = False) -> Trainer:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        accel = "gpu" if torch.cuda.is_available() else "cpu"
        if system() == "Darwin":
            accel = "mps"
        devices = 1
        logger = False if pretrained else TensorBoardLogger(str(self.logdir), name=None)
        cbs = [] if pretrained else callbacks()
        return Trainer(
            logger=logger,
            callbacks=cbs,
            accelerator=accel,
            devices=devices,
            enable_checkpointing=not pretrained,
            max_epochs=self.max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
        )

    def _get_model_args(self) -> Mapping:
        return dict(
            dataset=self.dataset,
            log_version_dir=self.logdir,
            hps=self.hparams,
        )
