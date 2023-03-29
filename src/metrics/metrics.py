from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from typing import Any, List, Literal, Optional
from warnings import simplefilter

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.enumerables import (
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
)
from src.metrics.base import PairwiseMetric
from src.metrics.functional import (
    cramer_v,
    error_acc,
    error_consistency,
    error_phi,
    kappa,
    mean_acc,
    percent_agreement,
)
from src.metrics.total import TotalPairwiseMetric
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

#
# Prediction-based
#


class PairedMeanAcc(PairwiseMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer = mean_acc
        self.kwargs: Any = {}
        self.name = "acc"


class TotalPairedMeanAcc(TotalPairwiseMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer = mean_acc
        self.kwargs: Any = {}
        self.name = "acc_t"


class PercentAgreement(PairwiseMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer = percent_agreement
        self.kwargs: Any = {}
        self.name = "pa"


class TotalPercentAgreement(TotalPairwiseMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer = percent_agreement
        self.kwargs: Any = {}
        self.name = "pa_t"


class CramerV(PairwiseMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer = cramer_v
        self.kwargs: Any = {}
        self.name = "v"


class TotalCramerV(TotalPairwiseMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer = cramer_v
        self.kwargs: Any = {}
        self.name = "v_t"


class Kappa(PairwiseMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer = kappa
        self.kwargs: Any = {}
        self.name = "k"


class TotalKappa(TotalPairwiseMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer = kappa
        self.kwargs: Any = {}
        self.name = "k_t"


#
# Error-based
#


class ErrorConsistency(PairwiseMetric):
    def __init__(
        self,
        results: Optional[Results] = None,
        local_norm: bool = False,
        empty_unions: Literal["nan", "0", "1"] = "nan",
    ) -> None:
        super().__init__(results)
        self.kwargs = dict(local_norm=local_norm, empty_unions=empty_unions)
        loc = "l" if local_norm else "g"
        un = {"nan": "_NA", "1": "_1", "0": ""}[empty_unions]
        self.name = f"ec_{loc}{un}"
        self.computer = error_consistency


# Classic EC
class TotalErrorConsistency(TotalPairwiseMetric):
    def __init__(
        self,
        results: Optional[Results],
        local_norm: bool = False,
        empty_unions: Literal["nan", "0", "1"] = "nan",
    ) -> None:
        super().__init__(results)
        self.local_norm = local_norm
        self.empty_unions: Literal["nan", "0", "1"] = empty_unions
        self.kwargs = dict(local_norm=local_norm, empty_unions=empty_unions)
        self.computer = error_consistency
        loc = "l" if local_norm else "g"
        un = {"nan": "_NA", "1": "_1", "0": ""}[empty_unions]
        self.name = f"ec_{loc}{un}_t"


class ErrorAccuracy(PairwiseMetric):
    def __init__(
        self,
        results: Results,
    ) -> None:
        super().__init__(results)
        self.computer = error_acc
        self.kwargs: Any = {}
        self.name = "e_acc"


class TotalErrorAccuracy(TotalPairwiseMetric):
    def __init__(
        self,
        results: Results,
    ) -> None:
        super().__init__(results)
        self.computer = error_acc
        self.kwargs: Any = {}
        self.name = "e_acc_t"


class ErrorPhi(PairwiseMetric):
    def __init__(
        self,
        results: Results,
    ) -> None:
        super().__init__(results)
        self.computer = error_phi
        self.kwargs: Any = {}
        self.name = "e_phi"


class TotalErrorPhi(TotalPairwiseMetric):
    def __init__(
        self,
        results: Results,
    ) -> None:
        super().__init__(results)
        self.computer = error_phi
        self.kwargs: Any = {}
        self.name = "e_phi_t"


def get_describe(arr: ndarray) -> DataFrame:
    s = pd.Series(arr).describe().to_frame().T
    s.index = [0]  # type: ignore
    return s


def get_describes_df(
    arrs: List[ndarray], label: str, show_progress: bool = False
) -> DataFrame:
    # tested chunksize below, seems to be fastest
    desc = f"Describing {label}s"
    dfs = process_map(
        get_describe, arrs, desc=desc, disable=not show_progress, chunksize=5
    )
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


SUBSET_METRICS = [
    PairedMeanAcc,
    PercentAgreement,
    CramerV,
    Kappa,
]
SUBSET_ERROR_METRICS = [
    ErrorConsistency,
    ErrorAccuracy,
    ErrorPhi,
]
TOTAL_METRICS = [
    TotalPairedMeanAcc,
    TotalPercentAgreement,
    TotalCramerV,
    TotalKappa,
]
TOTAL_ERROR_METRICS = [
    TotalErrorConsistency,
    TotalErrorAccuracy,
    TotalErrorPhi,
]
ALL_SUBSET_METRICS = SUBSET_METRICS + SUBSET_ERROR_METRICS
ALL_TOTAL_METRICS = TOTAL_METRICS + TOTAL_ERROR_METRICS
ALL_METRICS = ALL_SUBSET_METRICS + ALL_TOTAL_METRICS
# below needed because they require args
EC_UNION_METRICS = [ErrorConsistency, TotalErrorConsistency]


if __name__ == "__main__":

    simplefilter("error")

    # results = Results.from_tar_gz(ROOT / "hperturb.tar", save_test=True)
    PRELIM_DIR = ROOT / "debug_logs/prelim"
    FORCE = False
    results = Results.from_cached(root=PRELIM_DIR)
    dfs = []
    for metric in tqdm(ALL_METRICS, desc="Computing metrics"):
        if metric in EC_UNION_METRICS:
            dfl = metric(results, local_norm=True, empty_unions="0").compute(force=FORCE)
            dfg = metric(results, local_norm=False, empty_unions="0").compute(force=FORCE)
            dfs.append(dfl)
            dfs.append(dfg)
        else:
            dfs.append(metric(results).compute(force=FORCE))

    df_init = dfs[0]
    on_cols = df_init.columns[:-3]
    drop_cols = df_init.columns[:-3].to_list() + df_init.columns[-2:].to_list()
    metrics = [df.drop(columns=drop_cols) for df in dfs[1:]]
    df = pd.concat([df_init, *metrics], axis=1)
    df.to_parquet(PRELIM_DIR / "all_computed_metrics.parquet")
    sys.exit()
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
