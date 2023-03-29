from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from typing import Any, List, Literal, Optional

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm.contrib.concurrent import process_map

from src.enumerables import (
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
)
from src.metrics.base.cclass import ConsistencyClassPairwiseErrorMetric
from src.metrics.base.total import TotalPairwiseErrorMetric
from src.metrics.functional import (
    _accuracy,
    pairwise_error_acc,
    pairwise_error_consistency,
)
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


# class RunMetric(ABC):
#     """Abstract base class for metrics that need access to run predictions"""

#     def __init__(self, results: Optional[Results] = None) -> None:
#         super().__init__()
#         self.results: Optional[Results] = results
#         self.computed: Optional[DataFrame] = None
#         self.computer: MetricComputer
#         self.name: str = "default"

#     @property
#     def cached(self) -> Path:
#         """Is property so inherited classes can use overridden name"""
#         if self.results is None:
#             root: Path = ROOT
#         else:
#             root = self.results.root or ROOT
#         return root / f"{self.name}.parquet"

#     def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
#         if self.cached.exists() and not force:
#             return pd.read_parquet(self.cached)

#         if self.computed is not None:
#             return self.computed
#         if self.results is None:
#             raise ValueError("Cannot compute metrics when `self.results` is None.")

#         args = list(zip(self.results.preds, self.results.targs))
#         if len(args) < 1:
#             raise RuntimeError(
#                 f"Insufficient predictions / targets to compute {self.name} metric."
#             )
#         vals = process_map(
#             self.computer,
#             args,
#             total=len(args),
#             desc=f"Computing {self.name}",
#             chunksize=10,
#             disable=not show_progress,
#         )
#         self.computed = Series(
#             data=vals, index=self.results.evaluators.index, dtype=float, name=self.name
#         ).to_frame()
#         self.computed.to_parquet(self.cached)
#         print(f"Saved computed {self.name} metric to {self.cached}")
#         return self.computed


class TotalErrorConsistency(TotalPairwiseErrorMetric):
    """The classic / original EC definition"""

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
        self.computer = pairwise_error_consistency
        loc = "l" if local_norm else "g"
        un = {"nan": "_NA", "1": "_1", "0": ""}[empty_unions]
        self.name = f"ec_{loc}{un}"

    @property
    def cached(self) -> Path:
        """Is property so inherited classes can use overridden name"""
        if self.results is None:
            root: Path = ROOT
        else:
            root = self.results.root or ROOT
        return root / f"{self.name}.parquet"


class TotalPairwiseErrorAcc(TotalPairwiseErrorMetric):
    """The average accuracy of pairwise errors"""

    def __init__(
        self,
        results: Results,
    ) -> None:
        super().__init__(results)
        self.computer = pairwise_error_acc
        self.kwargs: Any = {}
        self.name = "tp_acc"


class SubsetPairwiseErrorAcc(ConsistencyClassPairwiseErrorMetric):
    """The average accuracy of pairwise errors on the inconsistent set"""

    def __init__(
        self,
        results: Results,
    ) -> None:
        super().__init__(results)
        self.computer = pairwise_error_acc
        self.kwargs: Any = {}
        self.name = "cc_acc"


class SubsetErrorConsistency(ConsistencyClassPairwiseErrorMetric):
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
        self.name = f"cc_ec_{loc}{un}"
        self.computer = pairwise_error_consistency


def compute_effect_sizes(dummy_df: DataFrame) -> float:
    raise NotImplementedError()


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


if __name__ == "__main__":

    # results = Results.from_tar_gz(ROOT / "hperturb.tar", save_test=True)
    PRELIM_DIR = ROOT / "debug_logs/prelim"
    results = Results.from_cached(root=PRELIM_DIR)

    # sys.exit()
    df = TotalErrorConsistency(results, local_norm=False, empty_unions="0").compute(
        force=True
    )
    df = SubsetErrorConsistency(results, local_norm=True, empty_unions="0").compute(
        force=True
    )
    sys.exit()
    df = ECAcc(results, local_norm=False).compute()
    df = ECAcc(results, local_norm=True, empty_unions="0").compute()
    df = TotalErrorConsistency(results, local_norm=False).compute()
    df = TotalErrorConsistency(results, local_norm=True, empty_unions="0").compute()
    acc = Accuracy(results).compute()
    sys.exit()

    descs = []
    for name in [DatasetName.Anneal]:
        for kind in [
            ClassifierKind.XGBoost,
            ClassifierKind.SGD_SVM,
            ClassifierKind.SGD_LR,
            ClassifierKind.LightGBM,
        ]:
            for dat_pert in [
                DataPerturbation.DoubleNeighbor,
                DataPerturbation.FullNeighbor,
                DataPerturbation.RelPercent20,
                DataPerturbation.Percentile20,
                DataPerturbation.SigDigZero,
                None,
            ]:
                for hp_pert in [
                    HparamPerturbation.SigZero,
                    HparamPerturbation.RelPercent20,
                    HparamPerturbation.AbsPercent20,
                    None,
                ]:
                    for tdown in [None, 50, 75]:
                        res = results.select(
                            dsnames=[name],
                            classifier_kinds=[kind],
                            reductions=[None],
                            cont_perturb=[dat_pert],
                            cat_perturb=[0.0],
                            hp_perturb=[hp_pert],
                            train_downsample=[tdown],
                        )
                        acc = Accuracy(res)
                        desc = acc.compute().describe()
                        info = {
                            "data": name.name,
                            "classifier": kind.value,
                            "cont_pert": "None" if dat_pert is None else dat_pert.value,
                            "cat_pert": float("nan") if cat_pert is None else cat_pert,
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
