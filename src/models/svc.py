from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from math import ceil
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Type

import numpy as np
import skops.io as skio
from numpy import ndarray
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC

from src.dataset import Dataset
from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.svm import (
    ClassicSVMHparams,
    LinearSVMHparams,
    NystroemHparams,
    SGDLinearSVMHparams,
)
from src.models.model import ClassifierModel


class ClassicSVM(ClassifierModel):
    def __init__(
        self, hparams: ClassicSVMHparams, dataset: Dataset, logdir: Path
    ) -> None:
        super().__init__(hparams=hparams, logdir=logdir, dataset=dataset)
        self.kind: ClassifierKind = ClassifierKind.SVM
        self.hparams: ClassicSVMHparams
        self.model_cls: Type[SVC] = SVC
        self.model: SVC

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        return self.model.predict(X), y

    def _get_model_args(self) -> Dict[str, Any]:
        args = super()._get_model_args()
        if RuntimeClass.from_dataset(self.dataset.name) is not RuntimeClass.Fast:
            args["cache_size"] = 512  # type: ignore
        return args


class LinearSVCModel(ClassifierModel):
    def __init__(
        self, hparams: ClassicSVMHparams, dataset: Dataset, logdir: Path
    ) -> None:
        super().__init__(hparams=hparams, logdir=logdir, dataset=dataset)
        self.kind: ClassifierKind = ClassifierKind.LinearSVM
        self.hparams: LinearSVMHparams
        self.model_cls: Type[LinearSVC] = LinearSVC
        self.model: LinearSVC

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        return self.model.predict(X), y


class SGDLinearSVCModel(ClassifierModel):
    def __init__(
        self, hparams: ClassicSVMHparams, dataset: Dataset, logdir: Path
    ) -> None:
        super().__init__(hparams=hparams, logdir=logdir, dataset=dataset)
        self.kind: ClassifierKind = ClassifierKind.SGD_SVM
        self.hparams: SGDLinearSVMHparams
        self.model_cls: Type[SGDClassifier] = SGDClassifier
        self.model: SGDClassifier

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        return self.model.predict(X), y

    def _get_model_args(self) -> Mapping:
        args: Dict[str, Any] = super()._get_model_args()
        args["n_jobs"] = 1
        return args
        # if RuntimeClass.from_dataset(self.dataset.name) is not RuntimeClass.Fast:
        #     args["n_jobs"] = -1  # type: ignore
        # return args


class NystroemSVM(ClassifierModel):
    def __init__(
        self,
        hparams: NystroemHparams,
        dataset: Dataset,
        logdir: Path,
    ) -> None:
        super().__init__(hparams=hparams, logdir=logdir, dataset=dataset)
        self.hparams: NystroemHparams = hparams
        self.classifier = SGDClassifier(**self.hparams.sgd_dict())
        self.kernel_approximator: Nystroem

    def fit(self, X: ndarray, y: ndarray, save: bool = True) -> None:
        ny_args = self.hparams.ny_dict()
        n_components = ny_args["n_components"]
        if X.shape[1] < n_components:
            n_components = X.shape[1]
        # UserWarning: n_components > n_samples. This is not possible.
        # n_components was set to n_samples, which results in inefficient evaluation of the full kernel.
        if n_components >= X.shape[0]:
            n_components = ceil(X.shape[0] // 2)
        self.kernel_approximator = Nystroem(
            kernel="rbf",
            gamma=ny_args["gamma"],
            n_components=n_components,
        )
        Xt = self.kernel_approximator.fit_transform(X)
        self.classifier.fit(Xt, y)

        if save:
            self.to_skops(self.logdir)
        self.fitted = True

    def to_skops(self, root: Path) -> None:
        model_out = root / "model.skops"
        ny_out = root / "nystroem.skos"
        skio.dump(self.classifier, model_out)
        skio.dump(self.kernel_approximator, ny_out)

    def from_skops(self, root: Path) -> Tuple[SGDClassifier, Nystroem]:
        model_out = root / "model.skops"
        ny_out = root / "nystroem.skos"
        return skio.load(model_out, trusted=True), skio.load(ny_out, trusted=True)

    def load_fitted(self) -> None:
        self.classifier, self.kernel_approximator = self.from_skops(self.logdir)

    def predict(self, X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
        Xt = self.kernel_approximator.fit_transform(X)
        return np.ravel(self.classifier.predict(Xt)), y
