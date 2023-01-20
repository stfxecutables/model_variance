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

By contrast, `PredMetric`s always need to be able to access the full preds/targs.

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
    ...
    results = Results.from_test_cached()
    res1 = results.select(
        dsnames=[DatasetName.Anneal],
        classifier_kinds=[ClassifierKind.XGBoost],
        reductions=[None],
        cont_perturb=[DataPerturbation.SigDigZero],
        cat_perturb="all",
        hp_perturb=[HparamPerturbation.SigZero],
        train_downsample=[None],
    )
    res2 = results.select(
        dsnames=[DatasetName.Anneal],
        classifier_kinds=[ClassifierKind.XGBoost],
        reductions=[None],
        cont_perturb=[DataPerturbation.RelPercent10],
        cat_perturb="all",
        hp_perturb=[HparamPerturbation.RelPercent10],
        train_downsample=[None],
    )
    res3 = results.select(
        dsnames=[DatasetName.Anneal],
        classifier_kinds=[ClassifierKind.XGBoost],
        reductions=[None],
        cont_perturb=[DataPerturbation.HalfNeighbor],
        cat_perturb="all",
        hp_perturb=[None],
        train_downsample=[None],
    )
    acc1 = Accuracy(res1)
    acc2 = Accuracy(res2)
    acc3 = Accuracy(res3)

    acc1.compute()
    acc2.compute()
    acc3.compute()

    print("")
