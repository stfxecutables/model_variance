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
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm import tqdm

from src.metrics.functional import MetricComputer, _pairwise_default, inconsistent_set
from src.results import Results


class Metric(ABC):
    """Abstract base class for metrics that need indices of the consistency classes
    in order to be computed."""

    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__()
        self.results: Optional[Results] = results
        self.computed: Optional[DataFrame] = None
        self.computer: MetricComputer = _pairwise_default
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
    ) -> tuple[List[DataFrame], List[List[ndarray]], List[List[ndarray]]]:
        def ne(pred: ndarray, targ: ndarray) -> NDArray[np.bool_]:
            if pred.ndim == 2:  # softmax values
                return np.argmax(pred, axis=1) != targ  # type: ignore
            return pred != targ  # type: ignore

        if self.results is None:
            raise ValueError("Cannot compute ic_info when `self.results` is None.")

        preds = self.results.preds
        targs = self.results.targs
        dfs = self.rep_dfs
        paired_rep_dfs: List[DataFrame] = []
        rep_ic_preds: List[List[ndarray]] = []
        rep_ic_targs: List[List[ndarray]] = []
        df: DataFrame
        for df in dfs:
            idx = df.index.to_list()
            row = df.iloc[0].to_frame().T
            rep_ic_preds.append([preds[i] for i in idx])
            rep_ic_targs.append([targs[i] for i in idx])
            k = len(idx)
            N = k * (k - 1) / 2
            # don't use numpy.repeat, destroys dtypes
            paired_df = row.loc[row.index.repeat(N)].reset_index(drop=True)
            paired_rep_dfs.append(paired_df)
        return paired_rep_dfs, rep_ic_preds, rep_ic_targs

    @abstractmethod
    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        if self.cached.exists() and not force:
            return pd.read_parquet(self.cached)

        if self.computed is not None:
            return self.computed
        if self.results is None:
            raise ValueError("Cannot compute metrics when `self.results` is None.")

        raise NotImplementedError()


class PairwiseMetric(Metric):
    """Abstract base class for metrics that need indices of the consistency classes
    in order to be computed, and operate on run pairs within a repeat."""

    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)
        self.computer: MetricComputer = _pairwise_default

    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        if self.cached.exists() and not force:
            return pd.read_parquet(self.cached)

        if self.computed is not None:
            return self.computed
        if self.results is None:
            raise ValueError("Cannot compute metrics when `self.results` is None.")

        preds = self.results.preds
        targs = self.results.targs
        idxs = self.rep_inconsistent_idxs
        paired_rep_dfs, rep_ic_preds, rep_ic_targs = self.ic_info

        metrics: List[ndarray] = []
        for preds, targs, idx in tqdm(
            zip(rep_ic_preds, rep_ic_targs, idxs),
            desc=f"Computing {self.name}",
            disable=not show_progress,
        ):
            metrics.append(self.computer(preds=preds, targs=targs, idx=idx))

        supplemented = []
        for df, metric, idx, targs in zip(paired_rep_dfs, metrics, idxs, rep_ic_targs):
            df[self.name] = metric
            df["N_ic"] = len(idx)
            df["N_test"] = len(targs[0])
            supplemented.append(df)

        mega_df = pd.concat(supplemented, axis=0, ignore_index=True)
        self.computed = mega_df
        mega_df.to_parquet(self.cached)
        print(f"Saved computed metrics to {self.cached}")
        return mega_df
