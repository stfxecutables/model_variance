from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import pickle
import sys
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Type, TypeVar, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm

from src.archival import parse_tar_gz
from src.constants import TESTING_TEMP
from src.enumerables import (
    CatPerturbLevel,
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
)
from src.hparams.hparams import Hparams
from src.results import Results

RunComputer = Callable[[tuple[ndarray, ndarray]], float]

def _default(preds_targs: tuple[ndarray, ndarray]) -> float:
    raise NotImplementedError("Must implement a `computer`!")


def _accuracy(preds_targs: tuple[ndarray, ndarray]) -> float:
    preds, targs = preds_targs
    if preds.ndim == 2:  # MLP
        return float(np.mean(np.argmax(preds, axis=1) == targs))
    return float(np.mean(preds == targs))
