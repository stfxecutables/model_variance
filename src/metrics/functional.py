from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from itertools import combinations
from typing import Any, Callable, List, Literal, Protocol, Union

import numpy as np
from numba import njit, prange
from numpy import ndarray
from numpy.typing import NDArray
from scipy.stats.contingency import association, crosstab
from sklearn.metrics import cohen_kappa_score as _kappa
from sklearn.metrics import confusion_matrix as confusion


class MetricComputer(Protocol):
    def __call__(
        self,
        preds: List[ndarray],
        targs: List[ndarray],
        idx: NDArray[np.int64],
        **kwargs: Any,
    ) -> ndarray:
        ...


def _pairwise_default(
    preds: List[ndarray], targs: List[ndarray], idx: NDArray[np.int64], **kwargs: Any
) -> ndarray:
    raise NotImplementedError("Must implement a `computer`!")


def inconsistent_set(rep_preds: List[ndarray]) -> NDArray[np.int64]:
    idx = np.ones_like(rep_preds[0], dtype=np.bool_)
    for preds in rep_preds:
        idx &= preds == rep_preds[0]
    return np.where(~idx)[0]


def acc(y1: ndarray, y2: ndarray) -> float:
    return np.mean(y1 == y2)


def _cramer_v(y1: ndarray, y2: ndarray) -> float:
    if len(np.unique(y1)) == 1 or len(np.unique(y2)) == 1:
        # can't correlate constants...
        return float("nan")
    ct = crosstab(y1, y2)[1]
    return float(association(observed=ct, correction=False))


def _pearson_r(y1: ndarray, y2: ndarray) -> float:
    if len(np.unique(y1)) == 1 or len(np.unique(y2)) == 1:
        # can't correlate constants...
        return float("nan")
    return float(np.corrcoef(y1, y1)[0, 1])


def _cohen_kappa(y1: ndarray, y2: ndarray) -> float:
    if (y1 == y2).all():
        return float("nan")
    return _kappa(y1, y2)


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


def _pairwise_error_consistency(
    preds: List[ndarray],
    targs: List[ndarray],
    idx: NDArray[np.int64],
    empty_unions: Literal["nan", "0", "1"] = "nan",
    local_norm: bool = False,
    **kwargs: Any,
) -> ndarray:
    if len(idx) == 0:
        k = len(preds)
        N = k * (k - 1) // 2
        return np.full((N,), fill_value=np.nan)
    y_errs = [pred[idx] != targ[idx] for pred, targ in zip(preds, targs)]
    matrix = _ecs(np.array(y_errs), empty_unions, local_norm)
    return matrix[np.triu_indices_from(matrix, k=1)]


def _pairwise_mean_acc(
    preds: List[ndarray],
    targs: List[ndarray],
    idx: NDArray[np.int64],
    **kwargs: Any,
) -> ndarray:
    if len(idx) == 0:
        k = len(preds)
        N = k * (k - 1) // 2
        return np.full((N,), fill_value=np.nan)
    accs = []
    for i, pred in enumerate(preds):
        for j, targ in enumerate(targs):
            if i >= j:
                continue
            accs.append(np.mean(pred[idx] == targ[idx]))
    return np.array(accs)


def _pairwise_error_acc(
    preds: List[ndarray],
    targs: List[ndarray],
    idx: NDArray[np.int64],
    **kwargs: Any,
) -> ndarray:
    if len(idx) == 0:
        k = len(preds)
        N = k * (k - 1) // 2
        return np.full((N,), fill_value=np.nan)
    y_errs = [pred[idx] != targ[idx] for pred, targ in zip(preds, targs)]
    matrix = _error_accs(np.array(y_errs))
    return matrix[np.triu_indices_from(matrix, k=1)]


def _pairwise_percent_agreement(
    preds: List[ndarray],
    targs: List[ndarray],
    idx: NDArray[np.int64],
    **kwargs: Any,
) -> ndarray:
    if len(idx) == 0:
        k = len(preds)
        N = k * (k - 1) // 2
        return np.full((N,), fill_value=np.nan)
    ys = [pred[idx] for pred in preds]
    y_combs = list(combinations(ys, r=2))
    pas = [acc(*comb) for comb in y_combs]  # Percent Agreement
    return np.array(pas)


def _pairwise_cramer_v(
    preds: List[ndarray],
    targs: List[ndarray],
    idx: NDArray[np.int64],
    **kwargs: Any,
) -> ndarray:
    if len(idx) <= 1:  # can't get a correlation for single values
        k = len(preds)
        N = k * (k - 1) // 2
        return np.full((N,), fill_value=np.nan)
    ys = [pred[idx] for pred in preds]
    y_combs = list(combinations(ys, r=2))
    pas = [_cramer_v(*comb) for comb in y_combs]
    return np.array(pas)


def _pairwise_kappa(
    preds: List[ndarray],
    targs: List[ndarray],
    idx: NDArray[np.int64],
    **kwargs: Any,
) -> ndarray:
    if len(idx) <= 1:
        k = len(preds)
        N = k * (k - 1) // 2
        return np.full((N,), fill_value=np.nan)
    ys = [pred[idx] for pred in preds]
    y_combs = list(combinations(ys, r=2))
    pas = [_cohen_kappa(*comb) for comb in y_combs]
    return np.array(pas)


def _pairwise_error_phi(
    preds: List[ndarray],
    targs: List[ndarray],
    idx: NDArray[np.int64],
    **kwargs: Any,
) -> ndarray:
    # see https://en.wikipedia.org/wiki/Phi_coefficient, which is just
    # Pearson correlation of two binary / booleans
    if len(idx) <= 1:  # can't get a correlation for single values
        k = len(preds)
        N = k * (k - 1) // 2
        return np.full((N,), fill_value=np.nan)
    y_errs = [pred != targ for pred, targ in zip(preds, targs)]
    ys = [err[idx] for err in y_errs]
    y_combs = list(combinations(ys, r=2))
    pas = [_pearson_r(*comb) for comb in y_combs]
    return np.array(pas)


# Prediction-based
mean_acc: MetricComputer = _pairwise_mean_acc
percent_agreement: MetricComputer = _pairwise_percent_agreement
cramer_v: MetricComputer = _pairwise_cramer_v
kappa: MetricComputer = _pairwise_kappa

# Error-based
error_consistency: MetricComputer = _pairwise_error_consistency
error_acc: MetricComputer = _pairwise_error_acc
error_phi: MetricComputer = _pairwise_error_phi
