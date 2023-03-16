from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from numba import njit, prange
from numpy import ndarray
from numpy.typing import NDArray

from src.results import PredTarg

RunComputer = Callable[[tuple[ndarray, ndarray]], float]
RunPairComputer = Callable[[tuple[PredTarg, PredTarg]], float]


def _default(preds_targs: tuple[ndarray, ndarray]) -> float:
    raise NotImplementedError("Must implement a `computer`!")


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
def _ecs_accs(
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
        acc_i = 1 - np.mean(err_i)
        for j in range(L):
            err_j = y_errs[j]
            acc_j = 1 - np.mean(err_j)
            acc = np.sqrt(acc_i * acc_j)
            if i == j:
                # matrix[i, j] = acc
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
            score = acc * np.sum(err_i & err_j) / local_union
            matrix[i, j] = matrix[j, i] = np.sqrt(score)
    return matrix


def _ec(
    y_errs: NDArray[np.bool_],
    empty_unions: Literal["nan", "0", "1"] = "nan",
    local_norm: bool = False,
) -> ndarray:
    matrix = _ecs(y_errs, empty_unions, local_norm)
    return matrix[np.triu_indices_from(matrix, k=1)]


def _ec_acc(
    y_errs: NDArray[np.bool_],
    empty_unions: Literal["nan", "0", "1"] = "nan",
    local_norm: bool = False,
) -> ndarray:
    matrix = _ecs_accs(y_errs, empty_unions, local_norm)
    return matrix[np.triu_indices_from(matrix, k=1)]


def _accuracy(preds_targs: tuple[ndarray, ndarray]) -> float:
    preds, targs = preds_targs
    if preds.ndim == 2:  # MLP
        return float(np.mean(np.argmax(preds, axis=1) == targs))
    return float(np.mean(preds == targs))
