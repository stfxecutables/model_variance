from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import pickle
import sys
from abc import ABC, abstractmethod
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Type, TypeVar, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.archival import parse_tar_gz
from src.constants import TESTING_TEMP
from src.enumerables import (
    CatPerturbLevel,
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
)
from src.hparams.hparams import Hparams
from src.metrics.functional import RunComputer, _accuracy, _default, _ec, _ec_acc
from src.results import Results

"""We need to distinguish between metrics that rely on the computation of
already-existing metrics (e.g. RepMeanAcc, RealVarAcc rely on having Acc
computed for each run) and metrics which need access to the raw predictions
from each run (e.g. EC).

The idea is to first compute run summary metrics (e.g. accuracy, AUROC, f1)
and then augment these metric values into Results.evaluators as columns. Then
this augmented DataFrame (or just the columns) can be passed into metrics that
only need summary values.

By contrast, `PredMetric`s always need to be able to access the full preds/targs
and/or hparams.

Likewise, there can be metrics that summarize within or across repeats. E.g. EC
summarizes within a repeat, but we cou
"""


class RunMetric(ABC):
    """Abstract base class for metrics that need access to run predictions"""

    def __init__(self, results: Results) -> None:
        super().__init__()
        self.results: Results = results
        self.computed: Series | None = None
        self.computer: RunComputer = _default
        self.name: str = "default"

    def compute(self, show_progress: bool = False) -> Series:
        if self.computed is not None:
            return self.computed
        args = list(zip(self.results.preds, self.results.targs))
        if len(args) < 1:
            raise RuntimeError(
                f"Insufficient predictions / targets to compute {self.name} metric."
            )
        vals = process_map(
            self.computer,
            args,
            total=len(args),
            desc=f"Computing {self.name}",
            chunksize=10,
            disable=not show_progress,
        )
        self.computed = Series(
            data=vals, index=self.results.evaluators.index, dtype=float, name=self.name
        )
        return self.computed


class ErrorConsistency:
    def __init__(
        self,
        results: Results,
        local_norm: bool = False,
        empty_unions: Literal["nan", "0", "1"] = "nan",
    ) -> None:
        self.results = results
        self.local_norm = local_norm
        self.empty_unions: Literal["nan", "0", "1"] = empty_unions
        # self.computed: tuple[list[DataFrame], list[ndarray]] | None = None
        self.computed: DataFrame | None = None
        self.computer = _ec
        self.name = "ec"

    def compute(self, show_progress: bool = False) -> DataFrame:
        def ne(pred: ndarray, targ: ndarray) -> ndarray:
            if pred.ndim == 2:  # softmax values
                return np.argmax(pred, axis=1) != targ
            return pred != targ

        if self.computed is not None:
            return self.computed

        preds = self.results.preds
        targs = self.results.targs
        reps, dfs = self.results.repeat_dfs()
        lengths: list[int] = []
        all_errors: list[ndarray] = []
        rep_dfs: list[DataFrame] = []
        df: DataFrame
        for rep, df in zip(reps, dfs):
            lengths.append(len(df))
            idx = df.index.to_list()
            row = df.iloc[0].to_frame().T
            errors = np.array([ne(preds[i], targs[i]) for i in idx])
            k = len(errors)
            N = k * (k - 1) / 2
            # don't use numpy.repeat, destroys dtypes
            rep_df = row.loc[row.index.repeat(N)].reset_index(drop=True)
            rep_dfs.append(rep_df)
            all_errors.append(errors)

        ecs: list[ndarray] = []
        for errors in tqdm(all_errors, desc="Computing ECs", disable=not show_progress):
            ecs.append(
                self.computer(
                    y_errs=errors,
                    empty_unions=self.empty_unions,
                    local_norm=self.local_norm,
                )
            )

        supplemented = []
        for df, ec in zip(rep_dfs, ecs):
            df[self.name] = ec
            supplemented.append(df)

        mega_df = pd.concat(supplemented, axis=0, ignore_index=True)
        self.computed = mega_df
        return mega_df


class ECAcc(ErrorConsistency):
    def __init__(
        self,
        results: Results,
        local_norm: bool = False,
        empty_unions: Literal["nan", "0", "1"] = "nan",
    ) -> None:
        super().__init__(results, local_norm, empty_unions)
        self.computer = _ec_acc
        self.name = "ec_acc"


class PairwiseRepeatMetric(RunMetric):
    """Metrics that summarize a repeat by examining all possible run-pairs within
    that repeat. E.g. EC."""


class SummaryMetric(ABC):
    """Base class for metrics that need access only to run summary metrics"""

    def __init__(self, data: DataFrame) -> None:
        super().__init__()
        self.data = data


class RepeatSummaryMetric(SummaryMetric):
    """Metrics that summarize / reduce a repeat by using all RunMetric metrics."""


class PairwiseSummaryMetric(RepeatSummaryMetric):
    """Metrics that summarize a repeat by examining all possible run-pairs within
    that repeat. E.g. average pairwise accuracy difference, accuracy variance."""


class TotalMetric(ABC):
    """Metrics that summarize across multiple repeats. E.g. any reduction of
    any of the above metrics."""


class Accuracy(RunMetric):
    def __init__(self, results: Results) -> None:
        super().__init__(results)
        self.computer = _accuracy
        self.name = "acc"


def compute_effect_sizes(dummy_df: DataFrame) -> float:
    cols = df.columns


def get_describe(arr: ndarray) -> DataFrame:
    s = pd.Series(arr).describe().to_frame().T
    s.index = [0]
    return s


def get_describes_df(
    arrs: list[ndarray], label: str, show_progress: bool = False
) -> DataFrame:
    # tested chunksize below, seems to be fastest
    desc = f"Describing {label}s"
    dfs = process_map(
        get_describe, arrs, desc=desc, disable=not show_progress, chunksize=5
    )
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


if __name__ == "__main__":

    # results = Results.from_tar_gz(ROOT / "hperturb.tar", save_test=True)
    results = Results.from_test_cached()
    # sys.exit()
    df = ECAcc(results, local_norm=False).compute()
    out = ROOT / "repeat_ec_accs_global_norm.parquet"
    df.to_parquet(out)
    print(f"Saved EC_accs to {out}")

    df = ECAcc(results, local_norm=True, empty_unions="0").compute()
    out = ROOT / "repeat_ec_accs_local_norm0.parquet"
    df.to_parquet(out)
    print(f"Saved EC_accs to {out}")
    sys.exit()

    df = ErrorConsistency(results, local_norm=False).compute()
    out = ROOT / "repeat_ecs_global_norm.parquet"
    df.to_parquet(out)
    print(f"Saved ECs to {out}")

    df = ErrorConsistency(results, local_norm=True, empty_unions="0").compute()
    out = ROOT / "repeat_ecs_local_norm_0.parquet"
    df.to_parquet(out)
    print(f"Saved ECs to {out}")

    acc = Accuracy(results)
    accs = acc.compute(show_progress=True)
    df = results.evaluators
    df["acc"] = accs
    out = ROOT / "prelim_accs.parquet"
    df.to_parquet(out)
    print(f"Saved accs to {out}")
    sys.exit()

    descs = []
    for name in [DatasetName.Anneal, DatasetName.Vehicle]:
        for kind in [
            ClassifierKind.XGBoost,
            ClassifierKind.SGD_SVM,
            ClassifierKind.SGD_LR,
        ]:
            for dat_pert in [
                DataPerturbation.SigDigZero,
                DataPerturbation.HalfNeighbor,
                DataPerturbation.RelPercent10,
                None,
            ]:
                for cat_pert in [None, 0.1]:
                    for hp_pert in [
                        HparamPerturbation.SigZero,
                        HparamPerturbation.RelPercent10,
                        HparamPerturbation.AbsPercent10,
                        None,
                    ]:
                        for tdown in [None, 50, 75]:
                            res = results.select(
                                dsnames=[name],
                                classifier_kinds=[kind],
                                reductions=[None],
                                cont_perturb=[dat_pert],
                                cat_perturb=[cat_pert],
                                hp_perturb=[hp_pert],
                                train_downsample=[tdown],
                            )
                            acc = Accuracy(res)
                            desc = acc.compute().describe()
                            info = {
                                "data": name.name,
                                "classifier": kind.value,
                                "cont_pert": "None"
                                if dat_pert is None
                                else dat_pert.value,
                                "cat_pert": float("nan")
                                if cat_pert is None
                                else cat_pert,
                                "hp_pert": "None" if hp_pert is None else hp_pert.value,
                                "tdown": str(tdown),
                            }
                            info.update(desc.to_dict())  # type: ignore
                            df = pd.DataFrame(info, index=[0])
                            descs.append(df)
                            print(df)
    df = pd.concat(descs, axis=0, ignore_index=True)
    OUT = ROOT / "prelim_results.parquet"
    df.to_parquet(OUT)
    print(f"Saved prelim df results to {OUT}")
    print(df)

    df_orig = df.copy()
    df["range"] = df["max"] - df["min"]
    df.drop(
        columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max"], inplace=True
    )
    df.groupby(["data", "classifier"], group_keys=False).apply(
        lambda grp: pd.get_dummies(grp)
    ).corr("spearman")["range"].sort_values()
    sbn.catplot(
        data=df,
        row="data",
        col="classifier",
        x="range",
        y="hp_pert",
        hue="cont_pert",
        kind="box",
    )
    plt.show()
