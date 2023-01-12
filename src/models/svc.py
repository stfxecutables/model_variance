from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Any, Mapping, Type

from numpy import ndarray
from sklearn.svm import SVC

from src.dataset import Dataset
from src.enumerables import ClassifierKind, DatasetName, RuntimeClass
from src.hparams.svm import SVMHparams
from src.models.model import ClassifierModel


class SVCModel(ClassifierModel):
    def __init__(self, hparams: SVMHparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__(hparams=hparams, logdir=logdir, dataset=dataset)
        self.kind: ClassifierKind = ClassifierKind.SVM
        self.hparams: SVMHparams
        self.model_cls: Type[SVC] = SVC
        self.model: SVC

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        return self.model.predict(X), y

    def _get_model_args(self) -> Mapping:
        args = super()._get_model_args()
        if RuntimeClass.from_dataset(self.dataset.name) is not RuntimeClass.Fast:
            args["cache_size"] = 512
        return args
