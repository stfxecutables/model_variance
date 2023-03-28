from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, Union

import skops.io as skio
from numpy import ndarray
from numpy.random import Generator
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.dataset import Dataset
from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.hparams import Hparams
from src.models.torch_base import MLP, LogisticRegression
from src.serialize import SKOPable

ThirdPartyClassifierModel = Union[SVC, XGBClassifier, MLP, LogisticRegression]


class ClassifierModel(SKOPable):
    def __init__(self, hparams: Hparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__()
        self.kind: ClassifierKind
        self.hparams: Hparams = hparams
        self.dataset = dataset
        self.logdir: Path = Path(logdir)
        self.runtime = RuntimeClass.from_dataset(self.dataset.name)
        self.fitted: bool = False
        self.model_cls: Type[Any]
        self.model: Optional[ThirdPartyClassifierModel] = None

    def fit(self, X: ndarray, y: ndarray, save: bool = False) -> None:
        # constant (non-perturbable) constructor args and fit args

        fargs = (X, y)
        args = self._get_model_args()
        self.model = self.model_cls(**args)
        self.model.fit(*fargs)
        if save:
            self.to_skops(self.logdir)
        self.fitted = True

    def tune(
        self, X: ndarray, y: ndarray, rng: Optional[Generator], iteration: int
    ) -> None:
        ...

    @abstractmethod
    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        ...

    def load_fitted(self) -> None:
        if self.model is not None:
            raise RuntimeError("Classifer already has fitted internal model.")
        self.model = self.from_skops(self.logdir)

    def to_skops(self, root: Path) -> None:
        outfile = root / "model.skops"
        skio.dump(self.model, file=outfile)

    def from_skops(self, root: Path) -> ThirdPartyClassifierModel:
        outfile = root / "model.skops"
        return skio.load(file=outfile, trusted=True)

    def _get_model_args(self) -> Dict[str, Any]:
        hps = self.hparams.to_dict()
        if self.kind is ClassifierKind.XGBoost:
            n_jobs = 1 if self.runtime is RuntimeClass.Fast else -1
            cargs: Mapping = dict(n_jobs=n_jobs)
            return {**cargs, **hps}
        else:
            return hps
