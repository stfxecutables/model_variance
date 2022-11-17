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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal
from xgboost import XGBClassifier

from src.dataset import Dataset


def estimate_runtime_xgb(dataset: Dataset) -> DataFrame | None:
    try:
        start = time()
        df = dataset.data
        loadtime = time() - start

        X = df.drop(columns="__target").to_numpy()
        X = StandardScaler().fit_transform(X)
        y = df["__target"].to_numpy()
        y = LabelEncoder().fit_transform(y).astype(np.float64)

        xgb = XGBClassifier(enable_categorical=True, tree_method="hist", n_jobs=1)
        xgb.fit(X, y)
        elapsed = (time() - start) / 60
        return DataFrame(
            {"classifier": "xgb_hist", "load_secs": loadtime, "fit_minutes": elapsed},
            index=[0],
        )
    except Exception:
        print_exc()
        print(f"Got error: {e}")
        return None



if __name__ == "__main__":
    datasets = Dataset.load_all()
    runtimes_xgb = process_map(estimate_runtime_xgb, desc="Timing XGBoost", max_workers=len(datasets))
    runtimes_xgb = [r for r in runtimes_xgb if r is not None]
    runtimes = pd.concat(runtimes_xgb, ignore_index=True, axis=0)
    outfile = ROOT / "xgb_hist_runtimes.json"
    runtimes.to_json(outfile)
    print(f"Saved runtimes DataFrame to {outfile}")