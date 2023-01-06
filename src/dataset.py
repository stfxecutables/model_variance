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
    Literal,
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
from numpy.random import Generator
from numpy.typing import NDArray
from pandas import CategoricalDtype, DataFrame, Series
from pandas.errors import PerformanceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.constants import CAT_REDUCED, CONT_REDUCED, DISTANCES, TEST_SIZE
from src.enumerables import DataPerturbation, DatasetName, RuntimeClass
from src.perturb import neighbour_perturb, sig_perturb_plus
from src.seeding import load_repeat_rng, load_run_rng

Percentage = Literal[25, 50, 75]
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

    def get_X_continuous(
        self,
        perturbation: DataPerturbation | None = None,
        reduction: int | None = None,
        rng: Generator | None = None,
    ) -> ndarray | None:
        if reduction not in [None, 25, 50, 75]:
            raise ValueError("`percent` must be in [None, 25, 50, 75]")
        if (reduction is not None) and (perturbation is not None):
            raise NotImplementedError("No plans to test perturbation on UMAP reductions.")

        rng = rng if rng is not None else np.random.default_rng()

        if reduction is not None:
            X: ndarray | None = reduce_continuous(self, percent=reduction)
            if X is None:
                return None
            if X.ndim == 0:
                X = X.reshape(-1, 1)
            return X

        df = self.data.drop(columns="__target")
        X_f: ndarray = np.asarray(
            df.select_dtypes(exclude=[CategoricalDtype]).astype(np.float64)
        )
        if X_f.shape[1] == 0:
            return None
        # NOTE: Neighbor perturbation *must* be done on the standardized data
        # because that is what it was calculated on. Sig-dig perturbation
        # also should be done on the standardized data (because of bad means)
        # and percentile or relative percent perturbation just shouldn't matter
        # if the data is standardized or not. Thus, we just work with the
        # standardized data in all cases.
        X_f -= X_f.mean(axis=0)
        X_f /= X_f.std(axis=0)

        if perturbation is None:
            return X_f
        if perturbation in [
            DataPerturbation.HalfNeighbor,
            DataPerturbation.QuarterNeighbor,
        ]:
            # we use the "normalized Gaussians" method to get a
            # random perturbation vector for each sample in d_nn / 2, where
            # d_nn = sqrt(|x^2 - x_nn^2|) for sample x and nearest neighor x_nn
            scale = 2 if perturbation is DataPerturbation.HalfNeighbor else 4
            distances = self.nearest_distances(reduction=reduction)
            perturbed = neighbour_perturb(X_f, distances=distances, scale=scale, rng=rng)
            return perturbed
        if perturbation in [DataPerturbation.SigDigOne, DataPerturbation.SigDigZero]:
            n_digits = 0 if perturbation is DataPerturbation.SigDigZero else 1
            perturbed = sig_perturb_plus(X_f, n_digits=n_digits, rng=rng)
            return perturbed
        if perturbation in [DataPerturbation.RelPercent05, DataPerturbation.RelPercent10]:
            magnitude = 0.05 if perturbation is DataPerturbation.RelPercent05 else 0.10
            deltas = rng.uniform(-magnitude, magnitude, size=X_f.shape)
            perturbed = X_f + deltas * X_f
            return perturbed
        if perturbation in [DataPerturbation.Percentile05, DataPerturbation.Percentile10]:
            """
            raise NotImplementedError(
                "This method doesn't make sense with clustered data. E.g. if you have\n"
                "data like:\n\n"
                "     .......                                               .....\n"
                "------^----------------------|------------------------------------>\n"
                "      |                     x=0\n"
                "      percentile\n"
                "then a 'small' negative perturbation is quite large. But if we take\n"
                "the absolute value, then I think we are OK"
            )
            """
            magnitude = 0.05 if perturbation is DataPerturbation.Percentile05 else 0.10
            # m = magnitude / 2
            percs = np.percentile(np.abs(X_f), q=magnitude, axis=0)
            # percs_lo = np.percentile(np.abs(X_f), q=m, axis=0)
            # we could have negatives in above
            deltas = [rng.uniform(-p, p, size=len(X_f)) for p in percs]
            deltas = np.stack(deltas, axis=1)
            perturbed = X_f + deltas * X_f
            return perturbed
        raise ValueError("Invalid Perturbation!")

    def get_X_categorical(
        self,
        perturbation_prob: float = 0,
        perturb_level: Literal["sample", "label"] = "label",
        reduction: int | None = None,
        rng: Generator | None = None,
    ) -> ndarray | None:
        """
        Parameters
        ----------
        perturbation_prob: float = 0
            If None, return data un-perturbed. Otherwise, return categoricals
            (NOT including the target) with "label noise" proportional to
            `perturbation_prob`. Note: this is

        perturb_level: "sample" | "label"
            If "label", simply perturb any label with probability
            `perturbation_prob`. When there are a lot of categorical features,
            this will mean most perturbed samples are different from the un-
            perturbed samples.

            If "sample", perturb `perturbation_prob` * len(self.data) samples
            only, with each label in a to-be-perturbed sample having a
            1 / n_categoricals probability of being perturbed (e.g. we expect
            only about 0-2 categorical values to be changed


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
        if reduction is not None and perturbation_prob > 0:
            raise ValueError(
                "Cannot perturb dimenion-reduced categoricals in this manner."
            )
        if perturbation_prob > 1:
            raise ValueError("`perturbation_prob` must be <= 1")

        if reduction is None:
            df = self.data.drop(columns="__target")
            cats = df.select_dtypes(include=[CategoricalDtype])
            if cats.shape[1] == 0:
                return OneHotEncoder(sparse=False).fit_transform(cats).astype(np.float64)  # type: ignore
            if perturbation_prob > 0:
                rng = np.random.default_rng() if rng is None else rng
                if perturb_level == "sample":
                    N = len(df)
                    ix = rng.permutation(N)[: ceil(perturbation_prob * N)]
                    ix_bool = np.zeros([len(cats)], dtype=np.bool_)
                    ix_bool[ix] = True

                    rands = cats.copy()  # df of random category values
                    for column in rands.columns:
                        c: Series = rands[column]
                        dtype: CategoricalDtype = c.dtype
                        categories = dtype.categories.to_numpy()
                        new = rng.choice(categories, size=len(c), replace=True)
                        rands[column] = new
                    n_categoricals = cats.shape[1]
                    p = 1 / n_categoricals
                    idx = rng.uniform(0, 1, size=cats.shape) < p
                    ix_final = ix_bool[:, None] & idx
                    cats.loc[ix_final] = rands[ix_final]
                else:
                    rands = cats.copy()
                    for column in cats.columns:
                        c: Series = cats[column]
                        dtype: CategoricalDtype = c.dtype
                        categories = dtype.categories.to_numpy()
                        new = rng.choice(categories, size=len(c), replace=True)
                        rands[column] = new
                    idx = rng.uniform(0, 1, size=rands.shape) < perturbation_prob
                    cats.loc[idx] = rands.loc[idx]
            return pd.get_dummies(cats).astype(np.float64).to_numpy()  # type: ignore

        X = reduce_categoricals(self)
        if X is None:  # fine, reduct is just continues vars
            return None
        if X.ndim == 0:
            X = X.reshape(-1, 1)
        return X

    def get_X_y(
        self,
        cont_perturb: DataPerturbation | None = None,
        cat_perturb_prob: float = 0,
        cat_perturb_level: Literal["sample", "label"] = "label",
        reduction: int | None = None,
        rng: Generator | None = None,
    ) -> tuple[ndarray, ndarray]:
        """
        Returns
        -------
        X: ndarray
            Predictors, perturbed depending on perturbation arguments.

        y: ndarray
            1d integer array of true labels. Never perturbed (no label noise).

        Note
        ----
        If `cont_perturb` is None, and `cat_perturb_prob == 0`, and
        reductions have already been generated, then a call to this
        function is fully deterministic and does NOT advance the
        passed in rng (or use it at all).
        """
        if rng is None:
            rng = np.random.default_rng()
        X_cat = self.get_X_categorical(
            perturbation_prob=cat_perturb_prob,
            perturb_level=cat_perturb_level,
            reduction=reduction,
            rng=rng,
        )
        X_cont = self.get_X_continuous(
            perturbation=cont_perturb,
            reduction=reduction,
            rng=rng,
        )
        y = self.data["__target"]
        y = LabelEncoder().fit_transform(y).astype(np.float64)
        X = np.concatenate([X_cat, X_cont], axis=1)
        return X, y

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
        if X is None or X.shape[1] == 0:
            return None

        if outfile.exists():
            return np.load(outfile).get("distances").astype(np.float64)

        cluster = str(os.environ.get("CC_CLUSTER")).lower()
        n_jobs = {"none": -1, "niagara": 40, "cedar": 32}[cluster]
        # if self.name is DatasetName.DevnagariScript:
        #     n_jobs = 6
        # if cluster is not None:
        #     os.environ["OPENBLAS_NUM_THREADS"] = str(n_jobs)

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
    def n_conts(self) -> int:
        df = self.data
        cat_dtypes = list(
            filter(
                lambda dtype: isinstance(dtype, CategoricalDtype),
                df.dtypes.unique().tolist(),
            )
        )
        return int(df.shape[1] - len(cat_dtypes) - 1)  # -1 for target

    @property
    def n_cont(self) -> int:
        return self.n_conts

    @property
    def n_continuous(self) -> int:
        return self.n_conts

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
        self,
        train_downsample: Percentage | None,
        cont_perturb: DataPerturbation | None = None,
        cat_perturb_prob: float = 0,
        cat_perturb_level: Literal["sample", "label"] = "label",
        reduction: int | None = None,
        repeat: int = 0,
        run: int = 0,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Returns
        -------
        X_train: ndarray
        y_train: ndarray
        X_test: ndarray
        y_test: ndarray
        """
        if train_downsample not in [None, 25, 50, 75]:
            raise ValueError("`train_downsample` must be in [None, 25, 50, 75]")
        shuffle_rng = load_repeat_rng(repeat=repeat)
        X_orig, y = self.get_X_y(
            cont_perturb=None,
            cat_perturb_prob=0,
            cat_perturb_level=cat_perturb_level,
            reduction=reduction,
            rng=shuffle_rng,  # rng is not advanced here since no perturbation
        )
        shuffle_idx = shuffle_rng.permutation(len(y))
        X_orig = X_orig[shuffle_idx]
        y = y[shuffle_idx]

        # 4-fold means test_size is 25%
        idx_train, idx_test = next(StratifiedKFold(4, shuffle=False).split(X_orig, y))
        X_test = X_orig[idx_test]
        y_test = y[idx_test]

        rng = load_run_rng(repeat=repeat, run=run)
        X_pert = self.get_X_y(
            cont_perturb=cont_perturb,
            cat_perturb_prob=cat_perturb_prob,
            cat_perturb_level=cat_perturb_level,
            reduction=reduction,
            rng=rng,
        )[0]
        X_pert = X_pert[shuffle_idx]
        X_nontest = X_pert[idx_train]
        y_nontest = y[idx_train]

        if train_downsample is None:
            X_train = X_nontest
            y_train = y_nontest
        else:
            N = len(X_nontest)
            idx = rng.permutation(N)
            X_nontest, y_nontest = X_nontest[idx], y_nontest[idx]
            # again need to go through k-fold for stratification...
            # Grab either train or test idx based on desired train size.
            # I.e. to get a 25% stratified train set, use 4-fold and take the
            # test split for training instead. To get a 75% stratified train
            # set, use the first normal train split of a 4-fold, and to get a
            # 50% train split, use the first split of a 2-fold.
            k = {25: 4, 50: 2, 75: 4}[train_downsample]
            split_idx = {25: 1, 50: 0, 75: 0}[train_downsample]
            idx_train = next(
                StratifiedKFold(k, shuffle=False).split(X_nontest, y_nontest)
            )[split_idx]
            X_train = X_nontest[idx_train]
            y_train = y_nontest[idx_train]
        return X_train, y_train, X_test, y_test

    @staticmethod
    def load_all() -> list[Dataset]:
        datasets: list[Dataset] = process_map(load, DatasetName, desc="Loading datasets")
        return datasets

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    ds = Dataset(DatasetName.Kr_vs_kp)
    for i, name in enumerate(DatasetName):
        ds = Dataset(name)
        if ds.n_continuous <= 0:
            continue
        print(ds.name.name)
        for _ in range(50):
            X = ds.get_X_continuous(
                perturbation=DataPerturbation.HalfNeighbor,
                reduction=None,
            )

    sys.exit()

    for i, name in enumerate(DatasetName):
        if i == 0:
            continue
        ds = Dataset(name)
        if ds.n_categoricals <= 1:
            continue
        print(ds.name.name)
        for _ in range(10):
            X = ds.get_X_categorical(perturbation_prob=0, reduction=None)
            X_cat = ds.get_X_categorical(
                perturbation_prob=0.1, perturb_level="label", reduction=None
            )
            X_cat2 = ds.get_X_categorical(
                perturbation_prob=0.1, perturb_level="sample", reduction=None
            )
            print(np.mean(X != X_cat))
            print(np.mean(X != X_cat2))
            # print(X)
            # print(X_cat)
        # break

    sys.exit()

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
