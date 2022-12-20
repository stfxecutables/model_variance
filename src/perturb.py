from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import time
from traceback import print_exc
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)
from warnings import catch_warnings, filterwarnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from numpy.typing import NDArray
from pandas import CategoricalDtype, DataFrame, Series
from pandas.errors import PerformanceWarning
from scipy.spatial import Voronoi
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal


def sig_perturb(x: ndarray, n_digits: int = 1) -> ndarray:
    delta = 10 ** (np.floor(np.log10(np.abs(x))) / 10 ** n_digits)
    return x + delta * np.random.uniform(-1, 1, x.shape)


def sig_perturb_plus(x: ndarray, n_digits: int = 1) -> ndarray:
    delta = 10 ** (np.floor(np.log10(np.abs(x))) / 10 ** n_digits)
    if n_digits == 1:
        delta *= 2
    return x + delta * np.random.uniform(-1, 1, x.shape)


def percent_perturb(x: float, xmin: float, xmax: float, magnitude: float) -> ndarray:
    """Perturb to within `magnitude` percent of the range of `x`"""
