from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from typing import Any, Callable, List, Literal, Protocol, Union

import numpy as np
from numba import njit, prange
from numpy import ndarray
from numpy.typing import NDArray
from scipy.stats.contingency import association, crosstab
from scipy.stats.distributions import uniform
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix as confusion

from src.results import PredTarg, PredTargIdx

RunPairComputer = Callable[[tuple[PredTarg, PredTarg]], float]


class PairwiseComputer(Protocol):
    def __call__(
        self,
        preds: List[ndarray],
        **kwargs: Any,
    ) -> ndarray:
        ...


class PairwiseErrorComputer(Protocol):
    def __call__(
        self,
        preds: List[ndarray],
        targs: List[ndarray],
        **kwargs: Any,
    ) -> ndarray:
        ...


class ConsistencyClassPairwiseComputer(Protocol):
    def __call__(
        self,
        preds: List[ndarray],
        targs: List[ndarray],
        idx: NDArray[np.int64],
        **kwargs: Any,
    ) -> ndarray:
        ...


class ConsistencyClassPairwiseErrorComputer(Protocol):
    def __call__(
        self,
        y_errs: NDArray[np.bool_],
        idx: NDArray[np.int64],
        **kwargs: Any,
    ) -> ndarray:
        ...


MetricComputer = Union[
    PairwiseComputer,
    PairwiseErrorComputer,
    ConsistencyClassPairwiseComputer,
    ConsistencyClassPairwiseErrorComputer,
]
ConsistencyClassComputer = Union[
    ConsistencyClassPairwiseComputer,
    ConsistencyClassPairwiseErrorComputer,
]


def _default(preds_targs: tuple[ndarray, ndarray]) -> float:
    raise NotImplementedError("Must implement a `computer`!")


def _cc_pairwise_default(
    preds: List[ndarray], targs: List[ndarray], idx: NDArray[np.int64], **kwargs: Any
) -> ndarray:
    raise NotImplementedError("Must implement a `computer`!")


def _cc_pairwise_error_default(
    y_errs: NDArray[np.bool_], idx: NDArray[np.int64], **kwargs: Any
) -> ndarray:
    raise NotImplementedError("Must implement a `computer`!")


def inconsistent_set(rep_preds: List[ndarray]) -> NDArray[np.int64]:
    idx = np.ones_like(rep_preds[0], dtype=np.bool_)
    for preds in rep_preds:
        idx &= preds == rep_preds[0]
    return np.where(~idx)[0]


def cramer_v(y1: ndarray, y2: ndarray) -> float:
    ct = crosstab(y1, y2).count
    return float(association(observed=ct, correction=False))


@njit(parallel=True)
def _ecs(
    y_errs: NDArray[np.bool_],
    empty_unions: Literal["nan", "0", "1"] = "nan",
    local_norm: bool = False,
) -> ndarray:
    """
    Parameters
    ----------
    y_errs: ndarray
        Array of shape (n_pairs, n_samples)
    """
    L = len(y_errs)
    # matrix = np.nan * np.ones((L, L))
    matrix = np.full((L, L), np.nan)
    for i in prange(L):
        err_i = y_errs[i]
        for j in range(L):
            err_j = y_errs[j]
            if i == j:
                matrix[i, j] = 1.0
                continue
            if i > j:
                continue
            if local_norm is True:
                union = err_i | err_j
                local_union = np.sum(union)
                if local_union == 0:
                    if empty_unions == "nan":
                        continue
                    elif empty_unions == "0":
                        matrix[i, j] = matrix[j, i] = 0.0
                        continue
                    elif empty_unions == "1":
                        matrix[i, j] = matrix[j, i] = 1.0
                        continue
            else:
                local_union = len(err_i)
            score = np.sum(err_i & err_j) / local_union
            matrix[i, j] = matrix[j, i] = score
    return matrix


@njit(parallel=True)
def _error_accs(y_errs: NDArray[np.bool_]) -> ndarray:
    """
    Parameters
    ----------
    y_errs: ndarray
        Array of shape (n_pairs, n_samples)
    """
    L = len(y_errs)
    # matrix = np.nan * np.ones((L, L))
    matrix = np.full((L, L), np.nan)
    for i in prange(L):
        err_i = y_errs[i]
        for j in range(i + 1, L):
            err_j = y_errs[j]
            ec = np.mean(err_i == err_j)
            matrix[i, j] = matrix[j, i] = ec
    return matrix


def _pairwise_accs(preds: List[ndarray], targs: List[ndarray], **kwargs: Any) -> ndarray:
    ...


# def _cramer_vs(

# )


def _ec(
    y_errs: NDArray[np.bool_],
    empty_unions: Literal["nan", "0", "1"] = "nan",
    local_norm: bool = False,
) -> ndarray:
    matrix = _ecs(y_errs, empty_unions, local_norm)
    return matrix[np.triu_indices_from(matrix, k=1)]


def _cc_ec(
    y_errs: NDArray[np.bool_],
    idx: NDArray[np.int64],
    empty_unions: Literal["nan", "0", "1"] = "nan",
    local_norm: bool = False,
    **kwargs: Any,
) -> ndarray:
    y_errs = y_errs[:, idx]  # y_ers.shape == (N_rep, n_samples)
    matrix = _ecs(y_errs, empty_unions, local_norm)
    return matrix[np.triu_indices_from(matrix, k=1)]


def _pairwise_error_acc(
    y_errs: NDArray[np.bool_], idx: NDArray[np.int64], **kwargs: Any
) -> ndarray:
    y_errs = y_errs[:, idx]  # y_ers.shape == (N_rep, n_samples)
    matrix = _error_accs(y_errs)
    return matrix[np.triu_indices_from(matrix, k=1)]


def _accuracy(preds_targs: tuple[ndarray, ndarray]) -> float:
    preds, targs = preds_targs
    if preds.ndim == 2:  # MLP
        return float(np.mean(np.argmax(preds, axis=1) == targs))
    return float(np.mean(preds == targs))
