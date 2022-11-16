from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

JSONS = ROOT / "data/json"

DROP_COLS = {
    "arrhythmia": ["J"],
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
MEDIAN_FILL_COLS: dict[str, list[str]] = {  # columns to fil lwith median column value
    "arrhythmia": ["T", "P", "QRST", "heartrate"],
}

UNKNOWN_FILL_COLS: dict[str, list[str]] = {  # categorical columns to make "unknown"
    "adult": ["workclass", "occupation", "native-country"],
}


DROP_ROWS = {
    "higgs": [98049],
}


def load(json: Path) -> Dataset:
    return Dataset(json)


class Dataset:
    """"""

    def __init__(self, path: Path) -> None:
        self.path = path
        stem = self.path.stem
        res = re.search(r"([0-9])_v([0-9]+)_(.*)", stem)
        if res is None:
            raise ValueError("Impossible", self.path)
        self.did = int(res[1])
        self.version = int(res[2])
        self.name = str(res[3]).lower()
        self.data_: Optional[DataFrame] = None

    @property
    def data(self) -> DataFrame:
        if self.data_ is None:
            df = pd.read_json(self.path)
            self.remove_nans(df)
            self.data_ = df
        return self.data_

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
                idx = df[col].isnull()
                df.loc[idx, col] = "unknown"
        if self.name in DROP_ROWS:
            df.drop(index=DROP_ROWS[self.name], inplace=True)

    def describe(self) -> None:
        df = self.data
        nrows, ncols = df.shape
        nan_targets = df["__target"].isnull().mean()
        nancol_counts = df.drop(columns=["__target"]).isnull().sum()
        nancols = nancol_counts[nancol_counts > 0]

        print("=" * 80)
        print(f"{self.name}")
        print("=" * 80)
        print(f"N_rows: {nrows}")
        print(f"N_cols: {ncols}")
        if nan_targets > 0:
            print(f"Percent NaN Targets: {np.round(nan_targets * 100, 1)}")
        if len(nancols > 0):
            print(f"NaNs: {nancol_counts.sum()}")
            print(f"{nancols}")
        print("")

    def get_monte_carlo_splits(
        self, train_size: int | float
    ) -> Tuple[DataFrame, DataFrame]:
        pass

    @staticmethod
    def load_all() -> List[Dataset]:
        jsons = sorted(JSONS.rglob("*.json"))
        return process_map(load, jsons, desc="Loading datasets")

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    datasets = Dataset.load_all()
    for dataset in datasets:
        if dataset.name not in ["higgs", "apsfailure"]:
            continue
        dataset.describe()