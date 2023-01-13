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
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC

from src.dataset import Dataset
from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.svm import LinearSVMHparams, SGDLinearSVMHparams, SVMHparams
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

    def _get_model_args(self) -> dict[str, Any]:
        args = super()._get_model_args()
        if RuntimeClass.from_dataset(self.dataset.name) is not RuntimeClass.Fast:
            args["cache_size"] = 512  # type: ignore
        return args


class LinearSVCModel(ClassifierModel):
    def __init__(self, hparams: SVMHparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__(hparams=hparams, logdir=logdir, dataset=dataset)
        self.kind: ClassifierKind = ClassifierKind.LinearSVM
        self.hparams: LinearSVMHparams
        self.model_cls: Type[LinearSVC] = LinearSVC
        self.model: LinearSVC

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        return self.model.predict(X), y


class SGDLinearSVCModel(ClassifierModel):
    def __init__(self, hparams: SVMHparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__(hparams=hparams, logdir=logdir, dataset=dataset)
        self.kind: ClassifierKind = ClassifierKind.SGD_SVM
        self.hparams: SGDLinearSVMHparams
        self.model_cls: Type[SGDClassifier] = SGDClassifier
        self.model: SGDClassifier

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        return self.model.predict(X), y

    def _get_model_args(self) -> Mapping:
        args: dict[str, Any] = super()._get_model_args()
        args["n_jobs"] = 1
        return args
        # if RuntimeClass.from_dataset(self.dataset.name) is not RuntimeClass.Fast:
        #     args["n_jobs"] = -1  # type: ignore
        # return args
