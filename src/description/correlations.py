from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import pickle
import re
import sys
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from seaborn import FacetGrid
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.archival import parse_tar_gz
from src.constants import PLOTS
from src.enumerables import (
    CatPerturbLevel,
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
)
from src.hparams.hparams import Hparams
from src.metrics.functional import RunComputer, _accuracy, _default
from src.results import Results

DROPS = [
    "count",
    "mean",
    "std",
    "min",
    "50%",
    "2.5%",
    "97.5%",
    "max",
    "range",
    "rrange",
]


def cleanup(df: DataFrame, metric: str, pairwise: bool = False) -> DataFrame:
    df = df.copy()
    remove = ["debug", "run"]
    if not pairwise:
        remove.append("repeat")
    for col in remove:
        if col in df.columns:
            df = df.drop(columns=col)
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(columns=col, inplace=True)
    if "categorical_perturb" in df.columns:
        df.categorical_perturb = df.categorical_perturb.fillna(0.0)
    if "train_downsample" in df.columns:
        df.train_downsample = df.train_downsample.fillna(100).apply(
            lambda x: f"{int(x):0d}%"
        )

    if pairwise:
        cols = df.columns.to_list()
        cols.remove("repeat")
        cols.remove(metric)
        print("Grouping repeats...")
        df = df.groupby(cols).describe(percentiles=[0.025, 0.5, 0.975])[metric]
    return df


def print_strongest_corrs(corrs: DataFrame, metric: str) -> DataFrame:
    df = corrs[metric].unstack().drop(columns=DROPS).reset_index()

    print("=" * 80)
    print(df.round(3))
    print(f"Strongest correlations with {metric}:")
    print("Excluding 'hp_sig-one':")
    idx = (
        df.drop(columns="hp_sig-one")
        .groupby(["data", "classifier"], group_keys=True)
        .apply(lambda g: g.abs())
        .idxmax(axis=1)
    )
    vals = [df[idx.values[i]][df[idx.values[i]].abs().idxmax()] for i in range(len(idx))]  # type: ignore
    frame = pd.concat(
        [idx.to_frame().reset_index().drop(columns="level_2"), pd.Series(vals)], axis=1
    )
    print(frame)

    print("Excluding 'hp_sig-one', 'hp_None':")
    idx = (
        df.drop(columns=["hp_sig-one", "hp_None"])
        .groupby(["data", "classifier"], group_keys=True)
        .apply(lambda g: g.abs())
        .idxmax(axis=1)
    )
    vals = [df[idx.values[i]][df[idx.values[i]].abs().idxmax()] for i in range(len(idx))]  # type: ignore
    frame = pd.concat(
        [idx.to_frame().reset_index().drop(columns="level_2"), pd.Series(vals)], axis=1
    )
    print(frame)


def print_acc_correlations() -> None:
    OUT = ROOT / "prelim_accs.parquet"
    df = pd.read_parquet(OUT)
    df = cleanup(df, metric="acc", pairwise=True).reset_index()
    df.rename(
        columns={
            "dataset_name": "data",
            "classifier_kind": "classifier",
            "continuous_perturb": "cp",
            "hparam_perturb": "hp",
        },
        inplace=True,
    )
    df["range"] = df["max"] = df["min"]
    df["rrange"] = df["97.5%"] - df["2.5%"]
    dummies = pd.get_dummies(df.loc[:, ["cp", "hp"]])
    df = pd.concat([df, dummies], axis=1).drop(columns=["cp", "hp"])
    corrs = df.groupby(["data", "classifier"]).corr()  # type: ignore

    print_strongest_corrs(corrs, "rrange")
    print_strongest_corrs(corrs, "range")
    print_strongest_corrs(corrs, "std")
    print_strongest_corrs(corrs, "mean")

    # corrs_mean = corrs["mean"].unstack().drop(columns=drops).round(3)
    # corrs_med = corrs["50%"].unstack().drop(columns=drops).round(3)
    # corrs_rrange = corrs["rrange"].unstack().drop(columns=drops).round(3)


def print_gross_descriptions() -> None:
    OUT = ROOT / "prelim_accs.parquet"
    EC_GLOBAL_OUT = ROOT / "repeat_ecs_global_norm.parquet"
    EC_LOCAL_OUT = ROOT / "repeat_ecs_local_norm_0.parquet"

    accs = pd.read_parquet(OUT)
    ecs = pd.read_parquet(EC_GLOBAL_OUT)
    els = pd.read_parquet(EC_LOCAL_OUT)
    accs = accs.loc[(accs.continuous_perturb == "None") & (accs.hparam_perturb == "None")]
    ecs = ecs.loc[(ecs.continuous_perturb == "None") & (ecs.hparam_perturb == "None")]
    els = els.loc[(els.continuous_perturb == "None") & (els.hparam_perturb == "None")]

    accs = cleanup(accs, metric="acc", pairwise=True)
    ecs = cleanup(ecs, metric="ec", pairwise=True)
    els = cleanup(els, metric="ec", pairwise=True)

    sep = "="*80
    print(sep)
    print("Accuracies")
    print(accs.round(3))
    print(sep)
    print("ECs (Dividing by Test Size)")
    print(ecs.round(3))
    print(sep)
    print("ECs (Dividing by Error Set Union)")
    print(els.round(3))
    print(sep)


if __name__ == "__main__":
    # cols are:
    #     "dataset_name", "classifier", "continuous_perturb",
    #     "categorical_perturb", "hparam_perturb",
    #     "train_downsample", "categorical_perturb_level", "acc",
    OUT = ROOT / "prelim_accs.parquet"
    EC_GLOBAL_OUT = ROOT / "repeat_ecs_global_norm.parquet"
    EC_LOCAL_OUT = ROOT / "repeat_ecs_local_norm_0.parquet"

    accs = pd.read_parquet(OUT)
    ecs = pd.read_parquet(EC_GLOBAL_OUT)
    els = pd.read_parquet(EC_LOCAL_OUT)

    # print_acc_correlations()
    print_gross_descriptions()

    # print("Overall average impact (correlation) of perturbation methods on accuracy")
    # print(
    #     accs.groupby(["dataset_name", "classifier_kind"], group_keys=True)
    #     .apply(lambda g: pd.get_dummies(g).corr("spearman")["acc"].sort_values())
    #     .unstack()
    #     .dropna(axis=1)
    #     .round(3)
    #     .describe()
    #     .T.sort_values(by="mean")
    # )
