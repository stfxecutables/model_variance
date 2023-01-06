from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.svm import SVC
from typing_extensions import Literal

from src.models.dl_model import DLModel
from src.models.torch_base import LogisticRegression


class LRModel(DLModel):
    def predict(self, X: ndarray) -> ndarray:
        model: LogisticRegression = self.model
        return model.predict(X)