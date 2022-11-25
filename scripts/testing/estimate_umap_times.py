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
from warnings import catch_warnings, filterwarnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from numpy.typing import NDArray
from pandas import CategoricalDtype, DataFrame, Series
from pandas.errors import PerformanceWarning
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal
from umap import UMAP
from xgboost import XGBClassifier

from src.constants import CAT_REDUCED
from src.dataset import Dataset
from src.enumerables import DatasetName, RuntimeClass

filterwarnings("ignore", category=PerformanceWarning)


def embed_categoricals(ds: Dataset) -> NDArray[np.float64] | None:
    """
    Notes
    -----
    We follow the guides:

        https://github.com/lmcinnes/umap/issues/58
        https://github.com/lmcinnes/umap/issues/104
        https://github.com/lmcinnes/umap/issues/241

    in spirit, but just embed all dummified categoricals to two dimensions.
    """
    outfile = CAT_REDUCED / f"{ds.name}.npy"
    if outfile.exists():
        reduced: NDArray = np.load(outfile)
        return reduced
    df = ds.data
    cats = df.select_dtypes(include=[CategoricalDtype])
    if cats.shape[1] == 0:
        return None

    x = pd.get_dummies(cats).astype(np.float64).to_numpy()
    umap = UMAP(n_components=2, metric="jaccard")
    with catch_warnings():
        filterwarnings("ignore", message="gradient function", category=UserWarning)
        reduced = umap.fit_transform(x)
    np.save(outfile, reduced)
    return reduced


def estimate_cat_embed_time(ds: Dataset) -> DataFrame | None:
    try:
        start = time()
        embed_categoricals(ds)
        embed_time = time() - start
        return DataFrame(
            {
                "dataset": ds.name,
                "n_samples": ds.n_samples,
                "n_cats": ds.n_cats,
                "embed_time": embed_time,
            },
            index=[0],
        )

    except Exception as e:
        print_exc()
        print(f"Got error: {e} for {ds.name}")
        return None

def compute_estimate_categorical_embedding_times() -> None:
    datasets = Dataset.load_all()
    outfiles = [
        ROOT / "cat_embed_fast_times.json",
        ROOT / "cat_embed_mid_times.json",
        ROOT / "cat_embed_slow_times.json",
    ]
    fast = [
        Dataset(name)
        for name in tqdm(RuntimeClass.Fast.members(), desc="Loading fast data")
    ]
    mid = [
        Dataset(name)
        for name in tqdm(RuntimeClass.Mid.members(), desc="Loading mid data")
    ]
    slow = [
        Dataset(name)
        for name in tqdm(RuntimeClass.Slow.members(), desc="Loading slow data")
    ]
    classes: list[list[Dataset]] = [fast, mid, slow]
    for compute_class, outfile in zip(classes, outfiles):
        runtimes = []
        if outfile.exists():
            continue
        desc = "Computing embeddings: {ds}"
        pbar = tqdm(compute_class, desc=desc.format(ds=""))
        for ds in pbar:
            if ds.name is DatasetName.Dionis:
                continue
            pbar.set_description(desc.format(ds=str(ds)))
            runtime = estimate_cat_embed_time(ds)
            if runtime is not None:
                runtimes.append(runtime)
        pbar.close()
        runtimes = pd.concat(runtimes, ignore_index=True, axis=0)
        runtimes.to_json(outfile)
        print(f"Saved runtimes DataFrame to {outfile}")


def get_float_X(df: DataFrame) -> DataFrame:
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
    if len(cat_dtypes) != 0:  # means some categoricals
        X_float = X.select_dtypes(exclude=cat_dtypes).astype(np.float64)
        X_float -= X_float.mean(axis=0)
        X_float /= X_float.std(axis=0)
        X_cat = X.select_dtypes(include=cat_dtypes)
        df = pd.concat([X_float, X_cat], axis=1).copy()
        return df

    # now must all be non-categorical
    df = X.astype(np.float64)
    return df


def estimate_runtime_xgb(dataset: Dataset) -> DataFrame | None:
    try:
        start = time()
        df = dataset.data
        loadtime = time() - start

        X = get_float_X(df)
        y = df["__target"]
        y = LabelEncoder().fit_transform(y).astype(np.float64)

        n_jobs = 1 if len(X) < 50000 else 80
        xgb = XGBClassifier(enable_categorical=True, tree_method="hist", n_jobs=n_jobs)
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
        get_float_X(df)
        y = df["__target"]
        LabelEncoder().fit_transform(y).astype(np.float64)
    except Exception as e:
        print_exc()
        print(f"Got error: {e} on dataset: {dataset.name} (id={dataset.did})")
        print(dataset)


if __name__ == "__main__":
    compute_estimate_categorical_embedding_times()
    sys.exit()

    outfile = ROOT / "xgb_hist_runtimes.json"
    fast = list(filter(lambda d: len(d) < 50000, datasets))
    slow = list(filter(lambda d: len(d) >= 50000, datasets))
    slow = sorted(slow, key=lambda d: -len(d))
    if outfile.exists():
        df = pd.read_json(outfile)
        df["data"] = list(map(lambda d: d.name, fast)) + list(map(lambda d: d.name, slow))
        df["n_cores"] = [1 for _ in fast] + [80 for _ in slow]
        df["mins_per_core"] = df["fit_minutes"] * df["n_cores"]
        df = df[
            ["data", "classifier", "load_secs", "fit_minutes", "n_cores", "mins_per_core"]
        ]
        df = df.sort_values(by=["mins_per_core", "fit_minutes"])
        df.to_json(outfile)
        print(df.to_markdown(tablefmt="simple"))
        sys.exit()

    runtimes_xgb_fast = process_map(
        estimate_runtime_xgb,
        fast,
        desc="Timing XGBoost: small datasets (< 50 000 samples)",
        max_workers=len(datasets),
    )
    runtimes_xgb_fast = [r for r in runtimes_xgb_fast if r is not None]
    runtimes_xgb_slow = list(
        map(
            estimate_runtime_xgb,
            tqdm(slow, desc="Timing XGBoost: large datasets (>= 50 000 samples)"),
        )
    )
    runtimes_xgb_slow = [r for r in runtimes_xgb_slow if r is not None]

    runtimes_xgb = runtimes_xgb_fast + runtimes_xgb_slow
    runtimes = pd.concat(runtimes_xgb, ignore_index=True, axis=0)
    runtimes.to_json(outfile)
    print(f"Saved runtimes DataFrame to {outfile}")