from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Type

from numpy import ndarray
from sklearn.svm import SVC

from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.hparams import Hparams
from src.models.model import ClassifierModel


class SVCModel(ClassifierModel):
    def __init__(self, hparams: Hparams, runtime: RuntimeClass) -> None:
        super().__init__(hparams, runtime)
        self.kind: ClassifierKind = ClassifierKind.SVM
        self.model_cls: Type[SVC] = SVC
        self.model: SVC

    def predict(self, X: ndarray, y: ndarray) -> ndarray:
        return self.model.predict(X)
