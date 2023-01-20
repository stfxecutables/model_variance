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
from src.metrics.functional import RunComputer, _accuracy, _default
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


class PredMetric(ABC):
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


class SummaryMetric(ABC):
    """Base class for metrics that need access only to run summary metrics"""

    def __init__(self, data: DataFrame) -> None:
        super().__init__()
        self.data = data


class RepeatSummaryMetric(SummaryMetric):
    """Metrics that summarize / reduce a repeat by using all runs."""


class PairwiseMetric(RepeatSummaryMetric):
    """Metrics that summarize a repeat by examining all possible run-pairs within
    that repeat. E.g. EC."""


class TotalMetric(ABC):
    """Metrics that summarize across multiple repeats. E.g. any reduction of
    any of the above metrics."""


class Accuracy(PredMetric):
    def __init__(self, results: Results) -> None:
        super().__init__(results)
        self.computer = _accuracy
        self.name = "acc"


if __name__ == "__main__":
    results = Results.from_test_cached()
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
