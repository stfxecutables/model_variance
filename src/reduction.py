from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from math import ceil
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import CategoricalDtype
from pandas.errors import PerformanceWarning
from sklearn.preprocessing import OneHotEncoder
from umap import UMAP

from src.constants import CAT_REDUCED, CONT_REDUCED
from src.dataset import Dataset
from src.enumerables import DatasetName

filterwarnings("ignore", category=PerformanceWarning)


def reduce_continuous(dataset: Dataset, percent: int) -> NDArray[np.float64] | None:
    # do not bother with all-categorical data here
    if dataset.name in [DatasetName.Kr_vs_kp, DatasetName.Car, DatasetName.Connect4]:
        return None
    outfile = CONT_REDUCED / f"{dataset.name.name}_{percent}percent.npy"
    if outfile.exists():
        reduced: NDArray = np.load(outfile)
        return reduced

    df = dataset.data.drop(columns="__target")
    X_float = df.select_dtypes(exclude=[CategoricalDtype]).astype(np.float64)
    X_float -= X_float.mean(axis=0)
    X_float /= X_float.std(axis=0)

    n_components = ceil((percent / 100) * X_float.shape[1])
    umap = UMAP(n_components=n_components)
    reduced = umap.fit_transform(X_float)
    np.save(outfile, reduced)
    return reduced


def reduce_categoricals(dataset: Dataset) -> NDArray[np.float64] | None:
    """
    Notes
    -----
    We follow the guides:

        https://github.com/lmcinnes/umap/issues/58
        https://github.com/lmcinnes/umap/issues/104
        https://github.com/lmcinnes/umap/issues/241

    in spirit, but just embed all dummified categoricals to two dimensions.
    """
    outfile = CAT_REDUCED / f"{dataset.name.name}.npy"
    if outfile.exists():
        reduced: NDArray[np.float64] = np.load(outfile)
        return reduced
    df = dataset.data.drop(columns="__target")
    cats = df.select_dtypes(include=[CategoricalDtype])
    if cats.shape[1] == 0:
        return OneHotEncoder().fit_transform(cats).astype(np.float64)  # type: ignore
    x = pd.get_dummies(cats).astype(np.float64).to_numpy()
    umap = UMAP(n_components=2, metric="jaccard")
    with catch_warnings():
        filterwarnings("ignore", message="gradient function", category=UserWarning)
        reduced = umap.fit_transform(x)
    np.save(outfile, reduced)
    return reduced
