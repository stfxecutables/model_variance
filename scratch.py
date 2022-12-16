from __future__ import annotations

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

from src.constants import RESULTS
from src.dataset import Dataset
from src.enumerables import DatasetName, RuntimeClass


def sig_perturb(x: ndarray, n_digits: int = 1) -> ndarray:
    delta = 10 ** (np.floor(np.log10(np.abs(x))) / 10 ** n_digits)
    return x + delta * np.random.uniform(-1, 1, x.shape)


def sig_perturb_demo() -> None:
    x = np.full([5000, 5000], np.random.uniform(0, 1000))
    print(f"Original x value: {x[0][0]:0.4e}")
    for N_DIGITS in [1, 2, 3]:
        pert = sig_perturb(x, n_digits=N_DIGITS)
        pmax = pert.max()
        pmin = pert.min()
        ps = np.array([pmin, pmax])
        np.set_printoptions(
            formatter={
                "float": lambda x: np.format_float_scientific(
                    x, precision=N_DIGITS, trim="k"
                )
            }
        )
        nth = {
            0: "zeroth",
            1: "first",
            2: "second",
            3: "third",
        }[N_DIGITS]
        print(f"Showing / perturbing at {nth} significant digit")
        print("              x:", x[0][0].reshape(-1, 1))
        # N2 = N_DIGITS if N_DIGITS > 1 else N_DIGITS + 1
        N2 = N_DIGITS
        np.set_printoptions(
            formatter={
                "float": lambda x: np.format_float_scientific(x, precision=N2, trim="k")
            }
        )
        print("perturbed range: ", ps)
        print("\n")
    sys.exit()


def plot_skewed_data() -> None:
    for dsname in tqdm([*DatasetName]):
        try:
            ds = Dataset(dsname)
            outdir = RESULTS / "plots/feature_dists"
            outdir.mkdir(exist_ok=True, parents=True)
            outfile = outdir / f"{ds.name.name}.png"
            X = ds.get_X_continuous(reduction=None)
            y = ds.data["__target"]
            fig, ax = plt.subplots(nrows=1, ncols=1)
            for i in range(X.shape[1]):
                x = X[:, i]
                sbn.kdeplot(x, ax=ax, lw=0.5)
                # ax.hist(x, bins=100)
            xmin, xmax = np.percentile(X.ravel(), [2.5, 97.5])
            ax.vlines([xmin, xmax], ymin=0, ymax=ax.get_ylim()[1])
            fig.suptitle(ds.name.name)
            fig.set_size_inches(w=15, h=10)
            fig.savefig(outfile, dpi=300)
            print(f"Saved figure to {outfile}")
            plt.close()
        except:
            print("Did not plot: ", ds.name.name)
            continue


if __name__ == "__main__":
    sig_perturb_demo()

    # for dsname in [
    #     DatasetName.ClickPrediction,
    #     DatasetName.CreditCardFraud,
    #     DatasetName.Dionis,
    #     DatasetName.Fabert,
    #     DatasetName.Miniboone,
    #     DatasetName.Shuttle,
    # ]:
    #     ds = Dataset(dsname)
    #     X = ds.get_X_continuous()
    #     print(f"\n{ds.name.name}")
    #     print(
    #         DataFrame(X)
    #         .describe(percentiles=[0.01, 0.025, 0.05, 0.50, 0.95, 0.975, 0.99])
    #         .T.sort_values(by="max", ascending=False)
    #     )
    #     print(f"Percent values >  5sd: {np.mean(X.ravel() > 5)}")
    #     print(f"Percent values > 10sd: {np.mean(X.ravel() > 10)}")

    # mega outliers:
    #   ClickPrediction
    #   CreditCardFraud
    #   Dionis
    #   Fabert
    #   Miniboone
    #   Shuttle

    # ds = Dataset(DatasetName.Arrhythmia)
    # dists = ds.nearest_distances(reduction=None)

    # df = ds.data
    # X = df.drop(columns="__target")
    # # below is about 5 minutes with 8 cores, 2.5 minutes on 40 cores
    # nn = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    # start = time()
    # nn.fit(X)
    # dists, neighb_idx = nn.kneighbors(X, n_neighbors=2, return_distance=True)
    # # zeros are just the points themselves
    # dists = dists[:, 1]
    # neighb_idx = neighb_idx[:, 1]
    # elapsed = time() - start
    # print(f"Elapsed second: {elapsed}")
    # print()