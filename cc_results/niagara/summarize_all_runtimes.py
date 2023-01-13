from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src.constants import CC_RESULTS


def set_long_print() -> None:
    pd.options.display.max_rows = 5000
    pd.options.display.max_info_rows = 5000
    pd.options.display.max_columns = 1000
    pd.options.display.max_info_columns = 1000
    pd.options.display.large_repr = "truncate"
    pd.options.display.expand_frame_repr = True
    pd.options.display.width = 200


def to_readable(duration_s: float) -> str:
    if duration_s <= 120:
        return f"{np.round(duration_s, 1):03.1f} sec"
    mins = duration_s / 60
    if mins <= 120:
        return f"{np.round(mins, 1):03.1f} min"
    hrs = mins / 60
    return f"{np.round(hrs, 2):03.2f} hrs"

def star_bool(value: bool | float) -> str:
    if str(value) == "NaN":
        return "NaN"
    if bool(value) is True:
        return "*"
    return ""



if __name__ == "__main__":
    set_long_print()

    times = CC_RESULTS
    jsons = sorted(times.rglob("*runtimes.json"))
    dfs = []
    for js in jsons:
        df = pd.read_json(js)
        df["classifier"] = js.name[: js.name.find("_")]
        df["cluster"] = js.parent.parent.parent.parent.name
        dfs.append(df)

    df_orig = pd.concat(dfs, axis=0, ignore_index=True)
    df = df_orig.groupby(["cluster", "classifier", "dataset"]).describe().sort_values(
        by=["cluster", "classifier", ("elapsed_s", "max")], ascending=False
    )


    runtimes = (
        df["elapsed_s"]  # type:ignore
        .drop(columns="count")
        .loc[:, ["min", "mean", "50%", "max", "std"]]
        .rename(columns={"50%": "med"})
        .applymap(to_readable)
    )
    accs = (
        df["acc"]  # type:ignore
        .sort_values(by="max", ascending=False)
        .drop(columns="count")
        .loc[:, ["min", "mean", "50%", "max"]]
        .rename(columns={"50%": "med"})
        .rename(columns=lambda s: f"acc_{s}")
    )
    accs["acc_range"] = accs["acc_max"] - accs["acc_min"]
    accs = accs.round(4)
    info = pd.concat([runtimes, accs], axis=1)
    print(info)
    svm = df_orig.loc[df_orig.classifier.isin(["svm-linear", "svm-sgd"])].drop(
        columns="elapsed_s"
    )
    svm = svm.loc[svm.cluster == "niagara"].drop(columns="cluster")
    svm = svm.groupby(["dataset", "classifier"]).describe()
    linear = (
        svm.query("classifier == 'svm-linear'")
        .reset_index()
        .drop(columns="classifier", level=0)
    )
    linear.index = linear["dataset"]
    linear.drop(columns="dataset", inplace=True, level=0)

    sgd = (
        svm.query("classifier == 'svm-sgd'")
        .reset_index()
        .drop(columns="classifier", level=0)
    )
    sgd.index = sgd["dataset"]
    sgd.drop(columns="dataset", inplace=True, level=0)

    diff = sgd["acc"] - linear["acc"]
    diff["sgm_mean_better"] = (diff["mean"] > 0).apply(star_bool)
    diff["sgm_max_better"] = (diff["max"] > 0).apply(star_bool)
    diff["sgm_min_better"] = (diff["min"] > 0).apply(star_bool)
    print(svm.unstack().round(4))
    print("sgd-svm - linear-svm acc differences (positive = sgd is better):")
    print(diff.round(5).dropna())
