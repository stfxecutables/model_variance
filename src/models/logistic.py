from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Any, Type

from numpy import ndarray
from sklearn.linear_model import SGDClassifier

from src.constants import LR_MAX_EPOCHS
from src.dataset import Dataset
from src.enumerables import ClassifierKind
from src.hparams.logistic import LRHparams, SGDLRHparams
from src.models.dl_model import DLModel
from src.models.model import ClassifierModel
from src.models.torch_base import LogisticRegression


class LRModel(DLModel):
    def __init__(self, hparams: LRHparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__(hparams=hparams, dataset=dataset, logdir=logdir)
        self.max_epochs = LR_MAX_EPOCHS
        self.kind: ClassifierKind = ClassifierKind.LR
        self.hparams: LRHparams
        self.model_cls: Type[LogisticRegression] = LogisticRegression
        self.model: LogisticRegression


class SGDLRModel(ClassifierModel):
    def __init__(self, hparams: LRHparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__(hparams=hparams, dataset=dataset, logdir=logdir)
        self.kind: ClassifierKind = ClassifierKind.SGD_LR
        self.hparams: SGDLRHparams
        self.model_cls: Type[SGDClassifier] = SGDClassifier
        self.model: SGDClassifier

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        return self.model.predict(X), y

    def _get_model_args(self) -> dict[str, Any]:
        args: dict[str, Any] = super()._get_model_args()
        args["n_jobs"] = 1
        return args
        # if RuntimeClass.from_dataset(self.dataset.name) is not RuntimeClass.Fast:
        #     args["n_jobs"] = -1  # type: ignore
        # return args
