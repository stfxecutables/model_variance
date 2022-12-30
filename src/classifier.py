from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from abc import ABC, abstractmethod
from typing import Any

from numpy import ndarray
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.enumerables import ClassifierKind, ClassifierModel
from src.hparams.hparams import Hparams
from src.hparams.svm import SVMHparams
from src.hparams.xgboost import XGBoostHparams


class Classifier(ABC):
    def __init__(self, kind: ClassifierKind, hparams: Hparams) -> None:
        self.kind: ClassifierKind = kind
        self.hparams: Hparams = hparams
        self.model: ClassifierModel = self.kind.model()

    @abstractmethod
    def fit(self, X: ndarray, y: ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: ndarray) -> None:
        ...
