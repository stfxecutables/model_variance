from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import pickle
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

CONT_PERTURB_ORDER = [
    "None",
    "sig0",
    "rel-10",
    "rel-20",
    "half-neighbour",
    "full-neighbour",
    "percentile-10",
    "percentile-20",
]
HP_ORDER = [
    "None",
    "sig-one",
    "sig-zero",
    "rel-percent-10",
    "rel-percent-20",
    "abs-percent-10",
    "abs-percent-20",
]
METRIC_BIGNAMES = {
    "acc": "Accuracy",
    "acc_range": "Accuracy Ranges",
    "ec": "Error Consistencies",
    "ec_mean": "Repeat Mean Error Consistencies",
    "ec_sd": "Repeat Error Consistency SDs",
}


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
        df = df.groupby(cols).describe()[metric]
    return df


def get_repeat_ranges(df: DataFrame) -> DataFrame:
    cols = [
        "dataset_name",
        "classifier_kind",
        "continuous_perturb",
        "categorical_perturb",
        "hparam_perturb",
        "train_downsample",
        "categorical_perturb_level",
    ]
    cols = [col for col in cols if col in df.columns]
    dfr = (df.groupby(cols)["acc"].max() - df.groupby(cols)["acc"].min()).reset_index()
    return dfr


def plot_rename(df: DataFrame) -> DataFrame:
    return df.rename(
        columns={
            "classifier_kind": "classifier",
            "categorical_perturb_level": "cat_perturb_level",
            "train_downsample": "train_size",
        }
    )


def fixup_plot(fig: Figure, dsname: str, metric: str, title_extra: str) -> None:
    metric_bigname = METRIC_BIGNAMES[metric]
    tex = "" if title_extra != "" else f" ({title_extra})"
    fig.suptitle(f"{metric_bigname} Distributions: data={dsname}{tex}")
    fig.tight_layout()
    # fig.set_size_inches(w=16, h=12)
    fig.set_size_inches(w=16, h=6)
    sbn.move_legend(fig, (0.89, 0.85))
    sbn.despine(fig, left=True, bottom=True)
    fig.subplots_adjust(top=0.92, left=0.1, bottom=0.075, right=0.9, hspace=0.2)


def violin_grid(
    df: DataFrame,
    dsname: str,
    metric: str,
    label: str,
    title_extra: str = "",
    show: bool = False,
) -> None:
    # hargs: dict[str, Any] = dict(hue="hparam_perturb", hue_order=HP_ORDER)
    hargs: dict[str, Any] = dict(hue="classifier")
    if "train_size" in df.columns:
        args = {**dict(row="train_size", row_order=["50%", "75%", "100%"]), **hargs}
    else:
        args = hargs
    sbn.catplot(
        data=df[df.dataset_name.isin([dsname])],
        # col="classifier",
        # x=metric,
        # y="continuous_perturb",
        y=metric,
        x="continuous_perturb",
        order=CONT_PERTURB_ORDER,
        # violin kwargs
        kind="violin",
        # bw="scott",
        bw=0.3,
        scale="area",
        cut=0.1,
        # bw="silverman",
        linewidth=0.5,
        #
        # box args
        # linewidth=0.5,
        # fliersize=0.5,
        **args,
    )
    fig = plt.gcf()
    metric_bigname = METRIC_BIGNAMES[metric]
    tex = "" if title_extra != "" else f" ({title_extra})"
    fig.suptitle(f"{metric_bigname} Distributions: data={dsname}{tex}")
    fig.tight_layout()
    # fig.set_size_inches(w=16, h=12)
    fig.set_size_inches(w=16, h=6)
    sbn.move_legend(fig, (0.89, 0.85))
    sbn.despine(fig, left=True, bottom=True)
    fig.subplots_adjust(top=0.92, left=0.1, bottom=0.075, right=0.9, hspace=0.2)
    if show:
        plt.show()
        plt.close()
        return

    out = PLOTS / f"{dsname}_{metric}s_{label}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot to {out}")


def strip_grid(
    df: DataFrame,
    dsname: str,
    metric: str,
    label: str,
    title_extra: str = "",
    show: bool = False,
) -> None:
    # hargs: dict[str, Any] = dict(hue="hparam_perturb", hue_order=HP_ORDER)
    hargs: dict[str, Any] = dict(hue="classifier")
    if "train_size" in df.columns:
        args = {**dict(row="train_size", row_order=["50%", "75%", "100%"]), **hargs}
    else:
        args = hargs
    sbn.catplot(
        data=df[df.dataset_name.isin([dsname])],
        # col="classifier",
        # x=metric,
        # y="continuous_perturb",
        y=metric,
        x="continuous_perturb",
        order=CONT_PERTURB_ORDER,
        # stripplot kwargs
        kind="strip",
        jitter=True,
        dodge=True,
        size=5.0,
        #
        # box args
        # linewidth=0.5,
        # fliersize=0.5,
        #
        **args,
    )
    fig = plt.gcf()
    metric_bigname = METRIC_BIGNAMES[metric]
    tex = "" if title_extra != "" else f" ({title_extra})"
    fig.suptitle(f"{metric_bigname} Distributions: data={dsname}{tex}")
    fig.tight_layout()
    # fig.set_size_inches(w=16, h=12)
    fig.set_size_inches(w=16, h=6)
    sbn.move_legend(fig, (0.89, 0.85))
    sbn.despine(fig, left=True, bottom=True)
    fig.subplots_adjust(top=0.92, left=0.1, bottom=0.075, right=0.9, hspace=0.2)
    if show:
        plt.show()
        plt.close()
        return

    out = PLOTS / f"{dsname}_{metric}s_{label}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot to {out}")


# def describe_ranges(df: DataFrame) -> DataFrame:
#     df["range"] = df["max"] - df["min"]

#     df.drop(
#         columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max"], inplace=True
#     )
#     df.groupby(["data", "classifier"], group_keys=False).apply(
#         lambda grp: pd.get_dummies(grp)
#     ).corr("spearman")["range"].sort_values()
#     raise NotImplementedError()


def plot_grid(
    df: DataFrame,
    dsname: str,
    kind: str,
    metric: str,
    label: str,
    title_extra: str,
    show: bool,
) -> None:
    if kind == "violin":
        violin_grid(
            df,
            dsname=dsname,
            metric=metric,
            label=label,
            title_extra=title_extra,
            show=show,
        )
    elif kind == "strip":
        strip_grid(
            df,
            dsname=dsname,
            metric=metric,
            label=label,
            title_extra=title_extra,
            show=show,
        )
    else:
        raise ValueError(f"Unrecognized kind: {kind}")


def plot_acc_dists(df: DataFrame, kind: str = "violin", show: bool = False) -> None:
    df = cleanup(df, metric="acc", pairwise=False)
    df = plot_rename(df)
    sbn.set_style("whitegrid")
    sbn.set_palette("pastel")
    for dsname in df.dataset_name.unique():
        plot_grid(
            df,
            dsname=dsname,
            kind=kind,
            metric="acc",
            label="",
            title_extra="",
            show=show,
        )


def plot_acc_ranges(df: DataFrame, kind: str = "violin", show: bool = False) -> None:
    df = cleanup(df, metric="acc", pairwise=False)
    df = get_repeat_ranges(df)
    df = plot_rename(df)
    df = df.rename(columns={"acc": "acc_range"})

    sbn.set_style("whitegrid")
    sbn.set_palette("pastel")
    for dsname in df.dataset_name.unique():
        plot_grid(
            df,
            dsname=dsname,
            kind=kind,
            metric="acc_range",
            label="",
            title_extra="",
            show=show,
        )


def plot_ec_dists(
    df: DataFrame, kind: str = "violin", local_norm: bool = False, show: bool = False
) -> None:
    df = cleanup(df, metric="ec", pairwise=False)
    df = plot_rename(df)
    sbn.set_style("whitegrid")
    sbn.set_palette("pastel")
    label = "local_norm_0" if local_norm else "global_norm"
    extra = "Local Norm" if local_norm else "Global Norm"

    for dsname in df.dataset_name.unique():
        plot_grid(
            df,
            dsname=dsname,
            kind=kind,
            metric="ec",
            label=label,
            title_extra=extra,
            show=show,
        )


def plot_ec_means(
    df: DataFrame, kind: str = "violin", local_norm: bool = False, show: bool = False
) -> None:
    df = cleanup(df, metric="ec", pairwise=True)
    df = df.reset_index().drop(
        columns=["count", "std", "min", "25%", "50%", "75%", "max"]
    )
    df = plot_rename(df)
    df.rename(columns={"mean": "ec_mean"}, inplace=True)
    sbn.set_style("whitegrid")
    sbn.set_palette("pastel")
    label = "local_norm_0" if local_norm else "global_norm"
    extra = "Local Norm" if local_norm else "Global Norm"

    for dsname in df.dataset_name.unique():
        plot_grid(
            df,
            dsname=dsname,
            kind=kind,
            metric="ec_mean",
            label=label,
            title_extra=extra,
            show=show,
        )


def plot_ec_sds(
    df: DataFrame, kind: str = "violin", local_norm: bool = False, show: bool = False
) -> None:
    df = cleanup(df, metric="ec", pairwise=True)
    df = df.reset_index().drop(
        columns=["count", "mean", "min", "25%", "50%", "75%", "max"]
    )
    df = plot_rename(df)
    df.rename(columns={"std": "ec_sd"}, inplace=True)
    sbn.set_style("whitegrid")
    sbn.set_palette("pastel")
    label = "local_norm_0" if local_norm else "global_norm"
    extra = "Local Norm" if local_norm else "Global Norm"

    for dsname in df.dataset_name.unique():
        plot_grid(
            df,
            dsname=dsname,
            kind=kind,
            metric="ec_sd",
            label=label,
            title_extra=extra,
            show=show,
        )


if __name__ == "__main__":
    # cols are:
    #     "dataset_name", "classifier", "continuous_perturb",
    #     "categorical_perturb", "hparam_perturb",
    #     "train_downsample", "categorical_perturb_level", "acc",
    OUT = ROOT / "prelim_accs.parquet"
    EC_GLOBAL_OUT = ROOT / "repeat_ecs_global_norm.parquet"
    EC_LOCAL_OUT = ROOT / "repeat_ecs_local_norm_0.parquet"

    KIND = "violin"
    SHOW = False
    # SHOW = True

    accs = pd.read_parquet(OUT)
    ecs = pd.read_parquet(EC_GLOBAL_OUT)
    els = pd.read_parquet(EC_LOCAL_OUT)

    plot_acc_dists(accs, show=SHOW)
    plot_acc_ranges(accs, show=SHOW)
    plot_ec_dists(ecs, local_norm=False, kind=KIND, show=SHOW)
    plot_ec_means(ecs, local_norm=False, kind=KIND, show=SHOW)
    plot_ec_sds(ecs, local_norm=False, kind=KIND, show=SHOW)

    plot_ec_dists(els, local_norm=True, kind=KIND, show=SHOW)
    plot_ec_means(els, local_norm=True, kind=KIND, show=SHOW)
    plot_ec_sds(els, local_norm=True, kind=KIND, show=SHOW)

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
