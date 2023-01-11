from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path
from time import time
from traceback import print_exc
from warnings import filterwarnings

import numpy as np
import pandas as pd
from pandas import CategoricalDtype, DataFrame
from pandas.errors import PerformanceWarning
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from xgboost import XGBClassifier

from src.dataset import Dataset

filterwarnings("ignore", category=PerformanceWarning)


def get_xgboost_X(df: DataFrame) -> DataFrame:
    X = df.drop(columns="__target").convert_dtypes(
        infer_objects=False,
        convert_string=True,
        convert_integer=False,
        convert_floating=True,  # type: ignore
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

        X = get_xgboost_X(df)
        y = df["__target"]
        y = LabelEncoder().fit_transform(y).astype(np.float64)  # type: ignore

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
        get_xgboost_X(df)
        y = df["__target"]
        LabelEncoder().fit_transform(y).astype(np.float64)  # type: ignore
    except Exception as e:
        print_exc()
        print(f"Got error: {e} on dataset: {dataset.name} (id={dataset.did})")
        print(dataset)


if __name__ == "__main__":
    outfile = ROOT / "xgb_hist_runtimes.json"
    datasets = Dataset.load_all()
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

    # process_map(check_conversions, datasets, desc="Checking conversions")
    # sys.exit()
    # runtimes_xgb = list(map(estimate_runtime_xgb, tqdm(datasets)))
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
