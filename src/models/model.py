from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Type

from numpy import ndarray
from numpy.random import Generator
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.dataset import Dataset
from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.hparams import Hparams
from src.models.torch_base import MLP, LogisticRegression

ThirdPartyClassifierModel = SVC | XGBClassifier | MLP | LogisticRegression


class ClassifierModel(ABC):
    def __init__(self, hparams: Hparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__()
        self.kind: ClassifierKind
        self.hparams: Hparams = hparams
        self.dataset = dataset
        self.logdir: Path = Path(logdir)
        self.runtime = RuntimeClass.from_dataset(self.dataset.name)
        self.fitted: bool = False
        self.model_cls: Type[Any]
        self.model: ThirdPartyClassifierModel | None

    def fit(self, X: ndarray, y: ndarray) -> None:
        # constant (non-perturbable) constructor args and fit args
        fargs = (X, y)
        args = self._get_model_args()
        self.model = self.model_cls(**args)
        self.model.fit(*fargs)
        self.fitted = True

    def tune(self, X: ndarray, y: ndarray, rng: Generator | None, iteration: int) -> None:
        ...

    @abstractmethod
    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        ...

    def _get_model_args(self) -> dict[str, Any]:
        hps = self.hparams.to_dict()
        if self.kind is ClassifierKind.XGBoost:
            n_jobs = 1 if self.runtime is RuntimeClass.Fast else -1
            cargs: Mapping = dict(n_jobs=n_jobs)
            return {**cargs, **hps}
        else:
            return hps
