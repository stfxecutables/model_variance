from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from abc import ABC, abstractmethod

from numpy import ndarray

from src.enumerables import ClassifierKind, ThridPartyClassifierModel
from src.hparams.hparams import Hparams


class Classifier(ABC):
    def __init__(self, kind: ClassifierKind, hparams: Hparams) -> None:
        self.kind: ClassifierKind = kind
        self.hparams: Hparams = hparams
        self.model: ThridPartyClassifierModel = self.kind.model()

    @abstractmethod
    def fit(self, X: ndarray, y: ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: ndarray) -> None:
        ...
