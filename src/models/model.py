from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import ABC, abstractclassmethod, abstractmethod, abstractstaticmethod
from pathlib import Path
from typing import (
    Any,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
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

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        if not self.fitted:
            raise RuntimeError("Model has not yet been fitted.")
        raise NotImplementedError("Subclass must implement `predict`")

    def _get_model_args(self) -> Mapping:
        hps = self.hparams.to_dict()
        if self.kind is ClassifierKind.XGBoost:
            n_jobs = -1 if self.runtime is RuntimeClass.Slow else 1
            cargs: Mapping = dict(n_jobs=n_jobs)
            return {**cargs, **hps}
        else:
            return hps
