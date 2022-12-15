from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import re
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from math import ceil
from pathlib import Path
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
from warnings import catch_warnings, filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from numpy.typing import NDArray
from pandas import CategoricalDtype, DataFrame, Series
from pandas.errors import PerformanceWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.constants import CAT_REDUCED, CONT_REDUCED, DISTANCES
from src.enumerables import DatasetName, RuntimeClass

JSONS = ROOT / "data/json"
PARQUETS = ROOT / "data/parquet"

DROP_COLS = {
    DatasetName.Arrhythmia: ["J"],
}

"""
For arrhythmia data:df.describe()[["T", "P", "QRST", "heartrate"]]
                T           P        QRST   heartrate
count  444.000000  430.000000  451.000000  451.000000
mean    36.150901   48.913953   36.716186   74.463415
std     57.858255   29.346409   36.020725   13.870684
min   -177.000000 -170.000000 -135.000000   44.000000
25%     14.000000   41.000000   12.000000   65.000000
50%     41.000000   56.000000   40.000000   72.000000
75%     63.250000   65.000000   62.000000   81.000000
max    179.000000  176.000000  166.000000  163.000000


"""
MEDIAN_FILL_COLS: dict[
    DatasetName, list[str]
] = {  # columns to fil lwith median column value
    DatasetName.Arrhythmia: ["T", "P", "QRST", "heartrate"],
}

UNKNOWN_FILL_COLS: dict[
    DatasetName, list[str]
] = {  # categorical columns to make "unknown"
    DatasetName.Adult: ["workclass", "occupation", "native-country"],
}


DROP_ROWS = {
    DatasetName.Higgs: [98049],
}


def load(name: DatasetName) -> Dataset:
    return Dataset(name)


def reduce_continuous(dataset: Dataset, percent: int) -> NDArray[np.float64] | None:
    # do not bother with all-categorical data here
    from umap import UMAP

    # import here to avoid container caching issue

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
    filterwarnings("ignore", category=PerformanceWarning)
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
    from umap import UMAP

    outfile = CAT_REDUCED / f"{dataset.name.name}.npy"
    if outfile.exists():
        reduced: NDArray[np.float64] = np.load(outfile)
        return reduced
    df = dataset.data.drop(columns="__target")
    cats = df.select_dtypes(include=[CategoricalDtype])
    if cats.shape[1] == 0:
        return OneHotEncoder(sparse=False).fit_transform(cats).astype(np.float64)  # type: ignore
    x = pd.get_dummies(cats).astype(np.float64).to_numpy()
    filterwarnings("ignore", category=PerformanceWarning)
    umap = UMAP(n_components=2, metric="jaccard")
    with catch_warnings():
        filterwarnings("ignore", message="gradient function", category=UserWarning)
        reduced = umap.fit_transform(x)
    np.save(outfile, reduced)
    return reduced


class Dataset:
    """"""

    def __init__(self, name: DatasetName) -> None:
        self.path = name.path()
        self.name = name
        stem = self.path.stem
        res = re.search(r"([0-9])_v([0-9]+)_(.*)", stem)
        if res is None:
            raise ValueError("Impossible", self.path)
        self.did = int(res[1])
        self.version = int(res[2])
        self.data_: Optional[DataFrame] = None

    @property
    def data(self) -> DataFrame:
        if self.data_ is None:
            # df = pd.read_json(self.path)
            df = pd.read_parquet(self.path)
            self.remove_nans(df)
            self.remove_constant(df)
            self.data_ = df
        return self.data_

    def get_X_continuous(self, reduction: int | None = None) -> ndarray | None:
        if reduction not in [None, 25, 50, 75]:
            raise ValueError("`percent` must be in [None, 25, 50, 75]")

        if reduction is None:
            df = self.data.drop(columns="__target")
            X_f: ndarray = np.asarray(
                df.select_dtypes(exclude=[CategoricalDtype]).astype(np.float64)
            )
            X_f -= X_f.mean(axis=0)
            X_f /= X_f.std(axis=0)
            return X_f

        X: ndarray | None = reduce_continuous(self, percent=reduction)
        if X is None:
            return None
        if X.ndim == 0:
            X = X.reshape(-1, 1)
        return X

    def get_X_categorical(self, reduction: int | None = None) -> ndarray | None:
        """
        Returns
        -------
        categoricals: ndarray
            If `reduction` is None, returns a concatenation of all one-hot
            encoded matrices of all categoricals, i.e. the equivalent of
            `pd.get_dummies(X_cat)` where X_cat is the categorical variables of
            the data.

            If `reduction` in [25, 50, 75], then returns the UMAP categorical
            reduction using the one-hot concatenation above, with the Jaccard
            distance metric, and thus the matrix is NOT sparsely populated as
            it is in the `None` case above.
        """
        if reduction not in [None, 25, 50, 75]:
            raise ValueError("`percent` must be in [None, 25, 50, 75]")

        if reduction is None:
            df = self.data.drop(columns="__target")
            cats = df.select_dtypes(include=[CategoricalDtype])
            if cats.shape[1] == 0:
                return OneHotEncoder(sparse=False).fit_transform(cats).astype(np.float64)  # type: ignore
            return pd.get_dummies(cats).astype(np.float64).to_numpy()  # type: ignore

        X = reduce_categoricals(self)
        if X is None:  # fine, reduct is just continues vars
            return None
        if X.ndim == 0:
            X = X.reshape(-1, 1)
        return X

    def X_reduced(self, percent: int) -> ndarray | None:
        if percent not in [25, 50, 75]:
            raise ValueError("`percent` must be in [25, 50, 75]")
        X_cont = reduce_continuous(self, percent)
        if X_cont is None:
            return None
        if X_cont.ndim == 0:
            X_cont = X_cont.reshape(-1, 1)

        X_cat = reduce_categoricals(self)
        if X_cat is None:  # fine, reduct is just continues vars
            return X_cont
        if X_cat.ndim == 0:
            X_cat = X_cat.reshape(-1, 1)
        X = np.concatenate([X_cat, X_cont], axis=1)
        return X

    def remove_nans(self, df: DataFrame) -> None:
        if self.name in DROP_COLS:
            df.drop(columns=DROP_COLS[self.name], inplace=True)
        if self.name in MEDIAN_FILL_COLS:
            cols = MEDIAN_FILL_COLS[self.name]
            medians = [df[col].median() for col in cols]
            for col, median in zip(cols, medians):
                idx = df[col].isnull()
                df.loc[idx, col] = median
        if self.name in UNKNOWN_FILL_COLS:
            cols = UNKNOWN_FILL_COLS[self.name]
            for col in cols:
                if isinstance(df[col].dtype, CategoricalDtype):
                    df[col].cat.add_categories("unknown").fillna("unknown")
                else:
                    idx = df[col].isnull()
                    df.loc[idx, col] = "unknown"
        if self.name in DROP_ROWS:
            df.drop(index=DROP_ROWS[self.name], inplace=True)

    def remove_constant(self, df: DataFrame) -> None:
        cont = df.select_dtypes(exclude=[CategoricalDtype])
        cols = cont.columns.to_numpy()
        sds = cont.std(axis=0).to_numpy()
        drop = cols[sds == 0]
        df.drop(columns=drop, inplace=True)

    def nearest_distances(self, reduction: int | None) -> ndarray | None:
        if reduction not in [None, 25, 50, 75]:
            raise ValueError("`percent` must be in [25, 50, 75]")
        label = "" if reduction is None else f"_reduce={int(reduction):02d}"
        outfile = DISTANCES / f"{self.name.name}{label}_distances.npz"

        X = self.get_X_continuous(reduction)
        if X is None:
            return None

        if outfile.exists():
            return np.load(outfile).get("distances").astype(np.float64)

        cluster = str(os.environ.get("CC_CLUSTER")).lower()
        n_jobs = {"none": -1, "niagara": 40, "cedar": 32}[cluster]
        if self.name is DatasetName.DevnagariScript:
            n_jobs //= 4
        if cluster is not None:
            os.environ["OPENBLAS_NUM_THREADS"] = str(n_jobs)

        nn = NearestNeighbors(n_neighbors=2, n_jobs=n_jobs)
        nn.fit(X)
        dists: ndarray = nn.kneighbors(X, n_neighbors=2, return_distance=True)[0][:, 1]
        np.savez_compressed(outfile, distances=dists.astype(np.float32))
        print(
            f"Saved precomputed distances for {self.name.name}@reduction={reduction} to {outfile}"
        )
        return dists

    @property
    def nrows(self) -> int:
        df = self.data
        return int(df.shape[0])

    @property
    def n_samples(self) -> int:
        return self.nrows

    @property
    def ncols(self) -> int:
        df = self.data
        return int(df.shape[1])

    @property
    def n_cats(self) -> int:
        df = self.data
        cat_dtypes = list(
            filter(
                lambda dtype: isinstance(dtype, CategoricalDtype),
                df.dtypes.unique().tolist(),
            )
        )
        return len(cat_dtypes)

    @property
    def n_categoricals(self) -> int:
        return self.n_cats

    def describe(self) -> None:
        df = self.data
        nrows, ncols = df.shape
        nan_targets = df["__target"].isnull().mean()
        nancol_counts = df.drop(columns=["__target"]).isnull().sum()
        nancols = nancol_counts[nancol_counts > 0]

        print("=" * 80)
        print(f"{self.name.name}")
        print("=" * 80)
        print(f"N_rows: {nrows}")
        print(f"N_cols: {ncols}")
        if nan_targets > 0:
            print(f"Percent NaN Targets: {np.round(nan_targets * 100, 1)}")
        if len(nancols > 0):
            print(f"NaNs: {nancol_counts.sum()}")
            print(f"{nancols}")
        print("")

    def __str__(self) -> str:
        df = self.data
        nrows, ncols = df.shape
        n_cat = self.n_cats
        return (
            f"{self.name.name}(N_rows: {nrows}, N_cols: {ncols}, "
            f"N_categoricals: {n_cat}/{100 * n_cat / ncols:0.1f}%)"
        )

    __repr__ = __str__

    def get_monte_carlo_splits(
        self, train_size: int | float
    ) -> Tuple[DataFrame, DataFrame]:
        pass

    @staticmethod
    def load_all() -> list[Dataset]:
        datasets: list[Dataset] = process_map(load, DatasetName, desc="Loading datasets")
        return datasets

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    datasets = Dataset.load_all()
    count = 0
    for dataset in datasets:
        print(dataset)
        x = dataset.X_reduced(25)
        if x is not None:
            print(x.shape)
            count += 1
        x = dataset.X_reduced(50)
        if x is not None:
            print(x.shape)
        x = dataset.X_reduced(75)
        if x is not None:
            print(x.shape)
    print(f"{count} usable reduced datasets.")  # 36