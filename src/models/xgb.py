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
from xgboost import XGBClassifier

from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.xgboost import XGBoostHparams
from src.models.model import ClassifierModel


class XGBoostModel(ClassifierModel):
    def __init__(
        self, hparams: XGBoostHparams, logdir: Path, runtime: RuntimeClass
    ) -> None:
        super().__init__(hparams=hparams, logdir=logdir, runtime=runtime)
        self.kind: ClassifierKind = ClassifierKind.XGBoost
        self.hparams: XGBoostHparams
        self.model_cls: Type[XGBClassifier] = XGBClassifier
        self.model: XGBClassifier

    def predict(self, X: ndarray, y: ndarray) -> ndarray:
        return self.model.predict(X)