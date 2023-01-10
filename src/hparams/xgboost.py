from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from abc import ABC
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src.hparams.hparams import (
    ContinuousHparam,
    FixedHparam,
    Hparam,
    Hparams,
    OrdinalHparam,
)


def xgboost_hparams(
    eta: float | None = None,
    lamda: float | None = None,
    alpha: float | None = None,
    num_round: int | None = None,
    gamma: float | None = None,
    colsample_bylevel: float | None = None,
    colsample_bynode: float | None = None,
    colsample_bytree: float | None = None,
    max_depth: int | None = None,
    max_delta_step: int | None = None,
    min_child_weight: float | None = None,
    subsample: float | None = None,
) -> list[Hparams]:
    """Note defaults are all None for XGBClassifier"""
    return [
        ContinuousHparam("eta", eta, max=1.0, min=0.001, log_scale=True),
        ContinuousHparam("lambda", lamda, max=1.0, min=1e-10, log_scale=True),
        ContinuousHparam("alpha", alpha, max=1.0, min=1e-10, log_scale=True),
        # XGB complains below are unused
        # OrdinalHparam("num_round", num_round, max=1000, min=1),
        ContinuousHparam("gamma", gamma, max=1.0, min=0.1, log_scale=True),
        ContinuousHparam(
            "colsample_bylevel", colsample_bylevel, max=1.0, min=0.1, log_scale=False
        ),
        ContinuousHparam(
            "colsample_bynode", colsample_bynode, max=1.0, min=0.1, log_scale=False
        ),
        ContinuousHparam(
            "colsample_bytree", colsample_bytree, max=1.0, min=0.1, log_scale=False
        ),
        OrdinalHparam("max_depth", max_depth, max=20, min=1),
        OrdinalHparam("max_delta_step", max_delta_step, max=10, min=0),
        ContinuousHparam(
            "min_child_weight", min_child_weight, max=20, min=0.1, log_scale=True
        ),
        ContinuousHparam("subsample", subsample, max=1.0, min=0.01, log_scale=False),
        FixedHparam("enable_categorical", value=True),
        FixedHparam("tree_method", value="hist"),
    ]


class XGBoostHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:

        if hparams is None:
            hparams = xgboost_hparams()
        super().__init__(hparams)


if __name__ == "__main__":
    xgbs = XGBoostHparams()
    print(xgbs)
    print(xgbs)
