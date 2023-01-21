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


def cleanup(df: DataFrame) -> DataFrame:
    for col in ["dimension_reduction", "debug", "run"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    df.categorical_perturb = df.categorical_perturb.fillna(0.0)
    df.train_downsample = df.train_downsample.fillna(100).apply(lambda x: f"{int(x):0d}%")
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


def describe_ranges(df: DataFrame) -> DataFrame:
    # df["range"] = df["max"] - df["min"]

    # df.drop(
    #     columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max"], inplace=True
    # )
    # df.groupby(["data", "classifier"], group_keys=False).apply(
    #     lambda grp: pd.get_dummies(grp)
    # ).corr("spearman")["range"].sort_values()
    raise NotImplementedError()


def plot_acc_dists(df: DataFrame) -> None:
    df = plot_rename(df)
    sbn.set_style("whitegrid")
    sbn.set_palette("pastel")
    for data in df.dataset_name.unique():
        sbn.catplot(
            data=df[df.dataset_name.isin([data])],
            row="train_size",
            row_order=["50%", "75%", "100%"],
            col="classifier",
            x="acc",
            y="continuous_perturb",
            order=["None", "half-neighbour", "rel-10", "sig0"],
            hue="hparam_perturb",
            # boxen args
            # kind="boxen",
            # linewidth=0.5,
            # showfliers=False,
            #
            # box args
            # linewidth=0.5,
            # fliersize=0.5,
            #
            # violin kwargs
            kind="violin",
            # bw=0.1,
            bw="scott",
            scale="area",
            # bw="silverman",
            linewidth=0.5,
            # cut=0,
            # saturation=0.5,
            # alpha=0.3,
            #
            # stripplot kwargs
            # kind="strip",
            # jitter=True,
            # dodge=True,
            # size=2.0,
        )
        fig = plt.gcf()
        fig.suptitle(f"Accuracy Distributions: data={data}")
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9, hspace=0.15, wspace=0.05)
        fig.set_size_inches(w=16, h=12)
        sbn.move_legend(fig, (0.89, 0.90))
        sbn.despine(fig, left=True)
        fig.subplots_adjust(top=0.92, left=0.1, bottom=0.075, right=0.9, hspace=0.2)
        out = PLOTS / f"{data}_accs.png"
        fig.savefig(out, dpi=150)
        plt.close()
        print(f"Saved plot to {out}")


def plot_acc_ranges(df: DataFrame) -> None:
    df = get_repeat_ranges(df)
    df = plot_rename(df)
    df = df.rename(columns={"acc": "acc_range"})

    sbn.set_style("whitegrid")
    sbn.set_palette("pastel")
    for data in df.dataset_name.unique():
        sbn.catplot(
            data=df[df.dataset_name.isin([data])],
            row="train_size",
            row_order=["50%", "75%", "100%"],
            col="classifier",
            x="acc_range",
            y="continuous_perturb",
            order=["None", "half-neighbour", "rel-10", "sig0"],
            hue="hparam_perturb",
            #
            # violin kwargs
            kind="violin",
            scale="width",
            bw="silverman",
            linewidth=0.5,
            # cut=0,
            #
            # stripplot kwargs
            # kind="strip",
            # jitter=True,
            # dodge=True,
            # size=2.0,
            #
            # box args
            # kind="box",
            # linewidth=0.5,
            # fliersize=0.5,
        )
        fig = plt.gcf()
        fig.suptitle(f"Distribution of Accuracy Ranges Across Repeats: data={data}")
        fig.tight_layout()
        fig.set_size_inches(w=16, h=12)
        sbn.move_legend(fig, (0.89, 0.90))
        sbn.despine(fig, left=True)
        fig.subplots_adjust(top=0.92, left=0.1, bottom=0.075, right=0.9, hspace=0.2)
        out = PLOTS / f"{data}_acc_ranges.png"
        fig.savefig(out, dpi=150)
        plt.close()
        print(f"Saved plot to {out}")


def plot_ec_dists(df: DataFrame, local_norm: bool) -> None:
    df = plot_rename(df)
    sbn.set_style("whitegrid")
    sbn.set_palette("pastel")
    for data in df.dataset_name.unique():
        sbn.catplot(
            data=df[df.dataset_name.isin([data])],
            row="train_size",
            row_order=["50%", "75%", "100%"],
            col="classifier",
            x="ec",
            y="continuous_perturb",
            order=["None", "half-neighbour", "rel-10", "sig0"],
            hue="hparam_perturb",
            # boxen args
            # kind="boxen",
            # linewidth=0.5,
            # showfliers=False,
            #
            # box args
            # linewidth=0.5,
            # fliersize=0.5,
            #
            # violin kwargs
            kind="violin",
            # bw=0.1,
            bw="scott",
            scale="area",
            # bw="silverman",
            linewidth=0.5,
            # cut=0,
            # saturation=0.5,
            # alpha=0.3,
            #
            # stripplot kwargs
            # kind="strip",
            # jitter=True,
            # dodge=True,
            # size=2.0,
        )
        fig = plt.gcf()
        label = "Local EC" if local_norm else "Global EC"
        fig.suptitle(f"EC Distributions: data={data} ({label})")
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9, hspace=0.15, wspace=0.05)
        fig.set_size_inches(w=16, h=12)
        sbn.move_legend(fig, (0.89, 0.90))
        sbn.despine(fig, left=True)
        fig.subplots_adjust(top=0.92, left=0.1, bottom=0.075, right=0.9, hspace=0.2)
        name = "local_norm_0" if local_norm else "global_norm"
        out = PLOTS / f"{data}_ecs_{name}.png"
        fig.savefig(out, dpi=150)
        plt.close()
        print(f"Saved plot to {out}")


if __name__ == "__main__":
    # cols are:
    #     "dataset_name", "classifier", "continuous_perturb",
    #     "categorical_perturb", "hparam_perturb",
    #     "train_downsample", "categorical_perturb_level", "acc",
    OUT = ROOT / "prelim_accs.parquet"
    EC_GLOBAL_OUT = ROOT / "repeat_ecs_global_norm.parquet"
    EC_LOCAL_OUT = ROOT / "repeat_ecs_local_norm_0.parquet"

    df = pd.read_parquet(EC_GLOBAL_OUT)
    df = cleanup(df)
    plot_ec_dists(df, local_norm=False)

    df = pd.read_parquet(EC_LOCAL_OUT)
    df = cleanup(df)
    plot_ec_dists(df, local_norm=True)

    df = pd.read_parquet(OUT)
    df = cleanup(df)
    plot_acc_dists(df)
    plot_acc_ranges(df)
