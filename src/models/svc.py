from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

from numpy import ndarray
from sklearn.svm import SVC

from src.models.model import ClassifierModel


class SVCModel(ClassifierModel):
    def predict(self, X: ndarray) -> ndarray:
        model: SVC = self.model
        return model.predict(X)
