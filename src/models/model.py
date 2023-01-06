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

from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.hparams import Hparams
from src.models.torch_base import MLP, LogisticRegression


class ClassifierModel(ABC):
    def __init__(self, hparams: Hparams, runtime: RuntimeClass) -> None:
        super().__init__()
        self.kind: ClassifierKind
        self.hparams: Hparams = hparams
        self.runtime = RuntimeClass(runtime)
        self.fitted: Any | None = None

        self.model = {
            ClassifierKind.LR: Type[LogisticRegression],
            ClassifierKind.MLP: Type[MLP],
            ClassifierKind.SVM: Type[SVC],
            ClassifierKind.XGBoost: Type[XGBClassifier],
        }[self.kind]

    def fit(self, X: ndarray, y: ndarray) -> None:
        cargs: Mapping
        n_jobs = -1 if self.runtime is RuntimeClass.Slow else 1
        # constant (non-perturbable) constructor args and fit args
        fargs = (X, y)
        if self.kind is ClassifierKind.XGBoost:
            cargs = dict(enable_categorical=True, tree_method="hist", n_jobs=n_jobs)
        elif self.kind is ClassifierKind.SVM:
            cargs = dict(kernel="rbf")
        elif self.kind is ClassifierKind.MLP:
            cargs = dict()
        elif self.kind is ClassifierKind.LR:
            cargs = dict()
        else:
            raise NotImplementedError()

        hps = self.hparams.to_dict()
        for key in cargs.keys():
            if key in hps.keys():
                raise ValueError(
                    f"Shared args over-riding hparams. Shared args:\n{cargs}\n"
                    f"hps:\n{hps}"
                )
        args = {**cargs, **hps}
        model = self.model(**args)
        model.fit(*fargs)
        self.fitted = model

    def predict(self, X: ndarray, y: ndarray) -> ndarray:
        if self.fitted is None:
            raise RuntimeError("Model has not yet been fitted.")
        raise NotImplementedError("Subclass must implement `predict`")
