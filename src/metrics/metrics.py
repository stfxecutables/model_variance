from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Series
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.enumerables import (
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
)
from src.metrics.functional import (
    ConsistencyClassComputer,
    ConsistencyClassPairwiseComputer,
    ConsistencyClassPairwiseErrorComputer,
    RunComputer,
    _accuracy,
    _cc_ec,
    _cc_pairwise_default,
    _cc_pairwise_error_default,
    _default,
    _ec,
    _ec_acc,
    inconsistent_set,
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


class RunMetric(ABC):
    """Abstract base class for metrics that need access to run predictions"""

    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__()
        self.results: Optional[Results] = results
        self.computed: Optional[DataFrame] = None
        self.computer: RunComputer = _default
        self.name: str = "default"

    @property
    def cached(self) -> Path:
        """Is property so inherited classes can use overridden name"""
        if self.results is None:
            root: Path = ROOT
        else:
            root = self.results.root or ROOT
        return root / f"{self.name}.parquet"

    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        if self.cached.exists() and not force:
            return pd.read_parquet(self.cached)

        if self.computed is not None:
            return self.computed
        if self.results is None:
            raise ValueError("Cannot compute metrics when `self.results` is None.")

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
        ).to_frame()
        self.computed.to_parquet(self.cached)
        print(f"Saved computed {self.name} metric to {self.cached}")
        return self.computed


class ConsistencyClassMetric(ABC):
    """Abstract base class for metrics that need indices of the consistency classes
    in order to be computed."""

    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__()
        self.results: Optional[Results] = results
        self.computed: Optional[DataFrame] = None
        self.computer: ConsistencyClassComputer = _cc_pairwise_default
        self.name: str = "default"

    @property
    def cached(self) -> Path:
        """Is property so inherited classes can use overridden name"""
        if self.results is None:
            root: Path = ROOT
        else:
            root = self.results.root or ROOT
        return root / f"repeat_{self.name}.parquet"

    @cached_property
    def rep_dfs(self) -> List[DataFrame]:
        if self.results is None:
            raise ValueError("Cannot compute metrics when `self.results` is None.")
        return self.results.repeat_dfs()[1]

    @cached_property
    def rep_inconsistent_idxs(self) -> List[NDArray[np.int64]]:
        if self.results is None:
            raise ValueError("Cannot compute metrics when `self.results` is None.")

        preds = self.results.preds
        dfs = self.rep_dfs

        # now we need to get the idx of the inconsistent set for each repeat's df
        idxs = []
        for df in dfs:
            output_idx = df.index.to_list()
            rep_preds = [preds[i] for i in output_idx]
            idx = inconsistent_set(rep_preds)
            idxs.append(idx)
        return idxs

    @cached_property
    def ic_info(
        self,
    ) -> tuple[
        List[NDArray[np.bool_]],
        List[DataFrame],
        List[List[ndarray]],
        List[List[ndarray]],
    ]:
        def ne(pred: ndarray, targ: ndarray) -> NDArray[np.bool_]:
            if pred.ndim == 2:  # softmax values
                return np.argmax(pred, axis=1) != targ  # type: ignore
            return pred != targ  # type: ignore

        preds = self.results.preds
        targs = self.results.targs
        dfs = self.rep_dfs
        idxs = self.rep_inconsistent_idxs
        all_errors: List[NDArray[np.bool_]] = []
        paired_rep_dfs: List[DataFrame] = []
        rep_ic_preds: List[List[ndarray]] = []
        rep_ic_targs: List[List[ndarray]] = []
        df: DataFrame
        for ic_idx, df in zip(idxs, dfs):
            idx = df.index.to_list()
            row = df.iloc[0].to_frame().T
            errors = np.array([ne(preds[i], targs[i]) for i in idx])
            rep_ic_preds.append([preds[i] for i in idx])
            rep_ic_targs.append([targs[i] for i in idx])
            k = len(errors)
            N = k * (k - 1) / 2
            # don't use numpy.repeat, destroys dtypes
            paired_df = row.loc[row.index.repeat(N)].reset_index(drop=True)
            paired_rep_dfs.append(paired_df)
            all_errors.append(errors)
        return all_errors, paired_rep_dfs, rep_ic_preds, rep_ic_targs

    @abstractmethod
    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        if self.cached.exists() and not force:
            return pd.read_parquet(self.cached)

        if self.computed is not None:
            return self.computed
        if self.results is None:
            raise ValueError("Cannot compute metrics when `self.results` is None.")

        preds = self.results.preds
        targs = self.results.targs
        dfs = self.rep_dfs
        idxs = self.rep_inconsistent_idxs
        all_errors, paired_rep_dfs, rep_ic_preds, rep_ic_targs = self.ic_info
        raise NotImplementedError()


class ConsistencyClassPairwiseMetric(ConsistencyClassMetric):
    """Abstract base class for metrics that need indices of the consistency classes
    in order to be computed, and operate on run pairs within a repeat."""

    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer: ConsistencyClassPairwiseComputer = _cc_pairwise_default

    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        if self.cached.exists() and not force:
            return pd.read_parquet(self.cached)

        if self.computed is not None:
            return self.computed
        if self.results is None:
            raise ValueError("Cannot compute metrics when `self.results` is None.")

        preds = self.results.preds
        targs = self.results.targs
        dfs = self.rep_dfs
        idxs = self.rep_inconsistent_idxs
        all_errors, paired_rep_dfs, rep_ic_preds, rep_ic_targs = self.ic_info

        metrics: List[ndarray] = []
        for preds, targs, idx in tqdm(
            zip(rep_ic_preds, rep_ic_targs, idxs),
            desc=f"Computing {self.name}",
            disable=not show_progress,
        ):
            metrics.append(self.computer(preds=preds, targs=targs, idx=idx))

        supplemented = []
        for df, metric in zip(paired_rep_dfs, metrics):
            df[self.name] = metric
            supplemented.append(df)

        mega_df = pd.concat(supplemented, axis=0, ignore_index=True)
        self.computed = mega_df
        mega_df.to_parquet(self.cached)
        print(f"Saved computed metrics to {self.cached}")
        return mega_df


class ConsistencyClassPairwiseErrorMetric(ConsistencyClassMetric):
    """Abstract base class for metrics that need indices of the consistency classes
    in order to be computed, and operate on run pairs within a repeat, and use the
    boolean / binary errors as metric basis."""

    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer: ConsistencyClassPairwiseErrorComputer = _cc_pairwise_error_default
        self.kwargs: Any

    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        if self.cached.exists() and not force:
            return pd.read_parquet(self.cached)

        if self.computed is not None:
            return self.computed
        if self.results is None:
            raise ValueError("Cannot compute metrics when `self.results` is None.")

        preds = self.results.preds
        targs = self.results.targs
        dfs = self.rep_dfs
        idxs = self.rep_inconsistent_idxs
        all_errors, paired_rep_dfs, rep_ic_preds, rep_ic_targs = self.ic_info

        metrics: List[ndarray] = []
        for errors, idx in tqdm(
            zip(all_errors, idxs),
            desc=f"Computing {self.name}",
            disable=not show_progress,
        ):
            metrics.append(self.computer(y_errs=errors, idx=idx, **self.kwargs))

        supplemented = []
        for df, metric, idx, targs in zip(paired_rep_dfs, metrics, idxs, rep_ic_targs):
            df[self.name] = metric
            df[f"{self.name}_N_ic"] = len(idx)
            df[f"{self.name}_N"] = len(targs[0])
            supplemented.append(df)

        mega_df = pd.concat(supplemented, axis=0, ignore_index=True)
        self.computed = mega_df
        mega_df.to_parquet(self.cached)
        print(f"Saved computed metrics to {self.cached}")
        return mega_df


class ErrorConsistency:
    def __init__(
        self,
        results: Optional[Results],
        local_norm: bool = False,
        empty_unions: Literal["nan", "0", "1"] = "nan",
    ) -> None:
        self.results = results
        self.local_norm = local_norm
        self.empty_unions: Literal["nan", "0", "1"] = empty_unions
        # self.computed: Optional[tuple[List[DataFrame], List[ndarray]]] = None
        self.computed: Optional[DataFrame] = None
        self.computer = _ec
        loc = "l" if local_norm else "g"
        un = {"nan": "_NA", "1": "_1", "0": ""}[empty_unions]
        self.name = f"ec{loc}{un}"

    @property
    def cached(self) -> Path:
        """Is property so inherited classes can use overridden name"""
        if self.results is None:
            root: Path = ROOT
        else:
            root = self.results.root or ROOT
        return root / f"{self.name}.parquet"

    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        if self.cached.exists() and not force:
            return pd.read_parquet(self.cached)

        def ne(pred: ndarray, targ: ndarray) -> ndarray:
            if pred.ndim == 2:  # softmax values
                return np.argmax(pred, axis=1) != targ  # type: ignore
            return pred != targ  # type: ignore

        if self.computed is not None:
            return self.computed
        if self.results is None:
            raise ValueError("Cannot compute metrics when `self.results` is None.")

        preds = self.results.preds
        targs = self.results.targs
        reps, dfs = self.results.repeat_dfs()
        lengths: List[int] = []
        all_errors: List[ndarray] = []
        rep_dfs: List[DataFrame] = []
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

        ecs: List[ndarray] = []
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
        mega_df.to_parquet(self.cached)
        print(f"Saved computed metrics to {self.cached}")
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
        loc = "l" if local_norm else "g"
        un = {"nan": "_NA", "1": "_1", "0": ""}[empty_unions]
        self.name = f"eca_{loc}{un}"


class CCErrorConsistency(ConsistencyClassPairwiseErrorMetric):
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
        self.computer = _cc_ec


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
    df = CCErrorConsistency(results, local_norm=True, empty_unions="0").compute(
        force=True
    )
    sys.exit()
    df = ECAcc(results, local_norm=False).compute()
    df = ECAcc(results, local_norm=True, empty_unions="0").compute()
    df = ErrorConsistency(results, local_norm=False).compute()
    df = ErrorConsistency(results, local_norm=True, empty_unions="0").compute()
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
