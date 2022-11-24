from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
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
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import CategoricalDtype, DataFrame, Series
from pandas.errors import PerformanceWarning
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal
from xgboost import XGBClassifier

from src.dataset import Dataset

filterwarnings("ignore", category=PerformanceWarning)


def get_xgboost_X(df: DataFrame) -> DataFrame:
    X = df.drop(columns="__target").convert_dtypes(
        infer_objects=False,
        convert_string=True,
        convert_integer=False,
        convert_floating=True,
    )
    cat_dtypes = list(
        filter(
            lambda dtype: isinstance(dtype, CategoricalDtype),
            X.dtypes.unique().tolist(),
        )
    )
    # TODO: Handle:
    # TypeError: Cannot setitem on a Categorical with a new category (unknown), set the categories first
    if len(cat_dtypes) != 0:  # means some categoricals
        X_float = X.select_dtypes(exclude=cat_dtypes).astype(np.float64)
        X_float -= X_float.mean(axis=0)
        X_float /= X_float.std(axis=0)
        X_cat = X.select_dtypes(include=cat_dtypes)
        df = pd.concat([X_float, X_cat], axis=1).copy()
        # From XGBoost: To get a de-fragmented frame, use `newframe = frame.copy()`
        return df

    # now must all be non-categorical
    df = X.astype(np.float64)
    return df


def estimate_runtime_xgb(dataset: Dataset) -> DataFrame | None:
    try:
        start = time()
        df = dataset.data
        loadtime = time() - start

        X = get_xgboost_X(df)
        y = df["__target"]
        y = LabelEncoder().fit_transform(y).astype(np.float64)

        xgb = XGBClassifier(enable_categorical=True, tree_method="hist", n_jobs=1)
        xgb.fit(X, y)
        elapsed = (time() - start) / 60
        return DataFrame(
            {"classifier": "xgb_hist", "load_secs": loadtime, "fit_minutes": elapsed},
            index=[0],
        )
    except Exception as e:
        print_exc()
        print(f"Got error: {e} on dataset: {dataset.name} (id={dataset.did})")
        print(dataset)
        return None


def check_conversions(dataset: Dataset) -> None:
    try:
        df = dataset.data
        get_xgboost_X(df)
        y = df["__target"]
        LabelEncoder().fit_transform(y).astype(np.float64)
    except Exception as e:
        print_exc()
        print(f"Got error: {e} on dataset: {dataset.name} (id={dataset.did})")
        print(dataset)


if __name__ == "__main__":
    datasets = Dataset.load_all()
    adult = list(filter(lambda d: d.name == "adult", datasets))[0]
    adult.data
    process_map(check_conversions, datasets, desc="Checking conversions")
    sys.exit()
    runtimes_xgb = list(map(estimate_runtime_xgb, tqdm(datasets)))
    # runtimes_xgb = process_map(
    #     estimate_runtime_xgb, datasets, desc="Timing XGBoost", max_workers=len(datasets)
    # )
    runtimes_xgb = [r for r in runtimes_xgb if r is not None]
    runtimes = pd.concat(runtimes_xgb, ignore_index=True, axis=0)
    outfile = ROOT / "xgb_hist_runtimes.json"
    runtimes.to_json(outfile)
    print(f"Saved runtimes DataFrame to {outfile}")