from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, List, Tuple, no_type_check

import numpy as np
import torch
from numpy import ndarray
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    CrossEntropyLoss,
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    Sequential,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

from src.constants import LR_MAX_EPOCHS, MLP_MAX_EPOCHS
from src.dataset import Dataset
from src.hparams.hparams import Hparams


class BaseModel(LightningModule):
    def __init__(
        self,
        dataset: Dataset,
        log_version_dir: Path,
        hps: Hparams,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model: Module
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.hps = Namespace(**hps.to_dict())
        self.max_epochs: int = LR_MAX_EPOCHS

        # task = "binary" if self.num_classes == 2 else "multiclass"
        acc_args: dict[str, Any] = dict(
            task="multiclass", num_classes=self.num_classes, top_k=1
        )
        self.train_acc = Accuracy(**acc_args)
        self.val_acc = Accuracy(**acc_args)
        self.test_acc = Accuracy(**acc_args)

        self.loss = CrossEntropyLoss()
        self.log_version_dir: Path = log_version_dir

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x

    @no_type_check
    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> Tensor:
        preds, loss, target = self._shared_step(batch)
        acc = self.train_acc(preds, target)
        self.log("train/loss", loss)
        self.log("train/acc", acc, prog_bar=True)
        if batch_idx % 20 == 0 and batch_idx != 0:
            self.train_acc(preds=preds, target=batch[1])
        return loss  # auto-logged by Lightning

    @no_type_check
    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> Any:
        pass
        # preds, loss, target = self._shared_step(batch)
        # if self.final_eval is not None:
        #     return {
        #         "pred": preds.cpu().numpy(),
        #         "target": batch[1].cpu().numpy(),
        #     }
        # self.val_metrics.log(self, preds, target)
        # self.log(f"{Phase.Val.value}/loss", loss, prog_bar=True)

    @no_type_check
    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> dict[str, ndarray]:
        preds, loss = self._shared_step(batch)[:2]
        self.test_acc(preds=preds, target=batch[1])
        self.log("test/loss", loss)
        return {
            "pred": preds.cpu().numpy(),
            "loss": loss.cpu().numpy(),
            "target": batch[1].cpu().numpy(),
        }

    @no_type_check
    def predict_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> dict[str, ndarray]:
        preds, loss = self._shared_step(batch)[:2]
        return {
            "pred": preds.cpu().numpy(),
            "loss": loss.cpu().numpy(),
            "target": batch[1].cpu().numpy(),
        }

    @no_type_check
    def test_epoch_end(self, outputs: list[dict[str, Tensor]]) -> dict[str, ndarray]:
        """Save predictions each epoch. We will compare to true values after."""
        if self.trainer is None:
            raise RuntimeError(f"LightningModule {self} has empty .trainer property")
        preds = [output["pred"] for output in outputs]
        targs = [output["target"] for output in outputs]
        losses = [output["loss"] for output in outputs]

        preds = np.concatenate(preds, axis=0)
        targs = np.concatenate(targs, axis=0)
        losses = np.ravel(losses)
        acc = accuracy(
            torch.from_numpy(preds),
            torch.from_numpy(targs),
            num_classes=self.num_classes,
            task="multiclass",
        )

        epoch = int(self.current_epoch)
        logdir = self.log_version_dir
        outdir = logdir / "test_preds"
        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir / f"test_preds_epoch={epoch:03d}.npy"
        np.save(outfile, preds)
        outfile = outdir / f"test_labels_epoch={epoch:03d}.npy"
        np.save(outfile, targs)
        outfile = outdir / f"test_acc={acc:0.4f}_epoch={epoch:03d}.npy"
        np.save(outfile, acc)
        return {
            "pred": preds,
            "target": targs,
        }

    def _shared_step(self, batch: Tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        x, target = batch
        preds = self(x)  # need pred.shape == (B, n_classes, H, W)
        loss = self.loss(preds, target)
        return preds, loss, target

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        opt = AdamW(
            self.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.wd,
        )
        sched = CosineAnnealingLR(
            optimizer=opt,
            T_max=self.max_epochs,
            eta_min=0,
        )
        return [opt], [sched]


class MlpBlock(Module):
    """Simple blocks of https://arxiv.org/pdf/1705.03098.pdf"""

    def __init__(self, width: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.block = Sequential(
            Linear(width, width, bias=False),
            BatchNorm1d(width),
            LeakyReLU(),
            Dropout(dropout),
            Linear(width, width, bias=False),
            BatchNorm1d(width),
            LeakyReLU(),
            Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return x + out


class BetterLinear(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = Sequential(
            Linear(in_channels, out_channels, bias=False),
            BatchNorm1d(out_channels),
            LeakyReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class MLP(BaseModel):
    """Similar to approach of https://arxiv.org/pdf/1705.03098.pdf"""

    def __init__(
        self,
        dataset: Dataset,
        log_version_dir: Path,
        hps: Hparams,
        width1: int = 512,
        width2: int = 256,
        dropout: float = 0.5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset=dataset, log_version_dir=log_version_dir, hps=hps, *args, **kwargs
        )
        self.in_channels = self.dataset.n_features
        self.model = Sequential(
            BetterLinear(in_channels=self.in_channels, out_channels=width1),
            MlpBlock(width=width1, dropout=dropout),
            BetterLinear(in_channels=width1, out_channels=width2),
            MlpBlock(width=width2, dropout=dropout),
            Linear(width2, self.num_classes, bias=True),
        )
        self.max_epochs = MLP_MAX_EPOCHS


class LogisticRegression(BaseModel):
    def __init__(
        self,
        dataset: Dataset,
        log_version_dir: Path,
        hps: Hparams,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset=dataset, log_version_dir=log_version_dir, hps=hps, *args, **kwargs
        )
        self.in_channels = self.dataset.n_features
        self.model = Linear(
            in_features=self.in_channels, out_features=self.num_classes, bias=True
        )
        self.max_epochs = LR_MAX_EPOCHS


if __name__ == "__main__":
    ...
