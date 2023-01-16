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
import seaborn as sbn
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


def compare_similar_models(model: Literal["svm", "lr"]) -> None:
    if model == "svm":
        select = ["svm-linear", "svm-sgd"]
        models = df_orig.loc[df_orig.classifier.isin(select)].drop(columns="elapsed_s")
        models = models.loc[models.cluster == "niagara"].drop(columns="cluster")
    else:
        select = ["lr", "lr-sgd"]
        models = df_orig.loc[df_orig.classifier.isin(select)].drop(columns="elapsed_s")
        sbn.catplot(
            models[models.classifier.apply(lambda s: "lr" in s)],
            y="acc",
            x="classifier",
            col="cluster",
            kind="violin",
        )
        plt.show()
        models = models.loc[
            ((models.cluster == "niagara") & (models.classifier == "lr-sgd"))
            | ((models.cluster == "cedar") & (models.classifier == "lr"))
        ].drop(columns="cluster")

    models = models.groupby(["dataset", "classifier"]).describe()
    non_sgd_model = (
        models.query(f"classifier == '{select[0]}'")
        .reset_index()
        .drop(columns="classifier", level=0)
    )
    non_sgd_model.index = non_sgd_model["dataset"]
    non_sgd_model.drop(columns="dataset", inplace=True, level=0)

    sgd_model = (
        models.query(f"classifier == '{select[1]}'")
        .reset_index()
        .drop(columns="classifier", level=0)
    )
    sgd_model.index = sgd_model["dataset"]
    sgd_model.drop(columns="dataset", inplace=True, level=0)

    diff = sgd_model["acc"] - non_sgd_model["acc"]
    diff["sgd_mean_better"] = (diff["mean"] > 0).apply(star_bool)
    diff["sgd_max_better"] = (diff["max"] > 0).apply(star_bool)
    diff["sgd_min_better"] = (diff["min"] > 0).apply(star_bool)
    print(models.unstack().round(4))
    print(f"{select[1]} - {select[0]} acc differences (positive = sgd is better):")
    print(diff.round(5).dropna())


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
    df = (
        df_orig.groupby(["cluster", "classifier", "dataset"])
        .describe()
        .sort_values(by=["cluster", "classifier", ("elapsed_s", "max")], ascending=False)
    )
    max_times = df["elapsed_s"]["max"].reset_index()
    max_times = max_times.loc[
        max_times.classifier.isin(["xgb", "lr-sgd", "svm-sgd", "mlp"])
    ]
    max_times = max_times.loc[
        ~((max_times.classifier == "mlp") & (max_times.cluster == "niagara"))
    ]

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
    print("")
    compare_similar_models(model="svm")
    print("")
    compare_similar_models(model="lr")
