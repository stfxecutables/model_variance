from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from functools import cached_property
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from src.metrics.base.cclass import (
    ConsistencyClassMetric,
    ConsistencyClassPairwiseErrorMetric,
    ConsistencyClassPairwiseMetric,
)
from src.results import Results


def get_all_idx(self: ConsistencyClassMetric) -> List[NDArray[np.int64]]:
    if self.results is None:
        raise ValueError("Cannot compute metrics when `self.results` is None.")

    preds = self.results.preds
    dfs = self.rep_dfs

    # inefficient, but saves code...
    idxs = []
    for df in dfs:
        output_idx = df.index.to_list()
        idx = np.arange(len(preds[output_idx[0]]), dtype=np.int64)
        idxs.append(idx)
    return idxs


class TotalMetric(ConsistencyClassMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)

    @cached_property
    def rep_inconsistent_idxs(self) -> List[NDArray[np.int64]]:
        return get_all_idx(self)

    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        df = super().compute(show_progress, force)
        return df.drop(columns="N_ic")


class TotalPairwiseMetric(ConsistencyClassPairwiseMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)

    @cached_property
    def rep_inconsistent_idxs(self) -> List[NDArray[np.int64]]:
        return get_all_idx(self)

    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        df = super().compute(show_progress, force)
        return df.drop(columns="N_ic")


class TotalPairwiseErrorMetric(ConsistencyClassPairwiseErrorMetric):
    def __init__(self, results: Optional[Results] = None) -> None:
        super().__init__(results)

    @cached_property
    def rep_inconsistent_idxs(self) -> List[NDArray[np.int64]]:
        return get_all_idx(self)

    def compute(self, show_progress: bool = False, force: bool = False) -> DataFrame:
        df = super().compute(show_progress, force)
        return df.drop(columns="N_ic")
