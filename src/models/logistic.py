from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Mapping, Type

from numpy import ndarray

from src.constants import LR_MAX_EPOCHS
from src.dataset import Dataset
from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.logistic import LRHparams
from src.models.dl_model import DLModel
from src.models.torch_base import LogisticRegression


class LRModel(DLModel):
    def __init__(self, hparams: LRHparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__(hparams=hparams, dataset=dataset, logdir=logdir)
        self.max_epochs = LR_MAX_EPOCHS
        self.kind: ClassifierKind = ClassifierKind.LR
        self.hparams: LRHparams
        self.model_cls: Type[LogisticRegression] = LogisticRegression
        self.model: LogisticRegression
