from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import ParameterGrid
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.enumerables import (
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
    RuntimeClass,
)
from src.evaluator import Evaluator
from src.hparams.hparams import Hparams
from src.hparams.logistic import SGDLRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import SGDLinearSVMHparams
from src.hparams.xgboost import XGBoostHparams


def create_grid() -> list[dict[str, Any]]:
    classifiers = [ClassifierKind.SGD_LR, ClassifierKind.SGD_SVM, ClassifierKind.XGBoost]
    data_perturbs = [
        None,
        DataPerturbation.HalfNeighbor,
        DataPerturbation.RelPercent10,
        DataPerturbation.SigDigZero,
    ]
    hp_perturbs = [
        None,
        HparamPerturbation.SigZero,
        HparamPerturbation.RelPercent10,
        HparamPerturbation.AbsPercent10,
    ]
    dsnames = RuntimeClass.most_fastest()
    grid = list(
        ParameterGrid(
            dict(
                dataset_name=dsnames,
                classifier_kind=classifiers,
                repeat=list(range(10)),
                run=list(range(10)),
                # hparams needs to be handled manually
                dimension_reduction=[None],
                continuous_perturb=data_perturbs,
                categorical_perturb=[None, 0.1],
                hparam_perturb=hp_perturbs,
                # train_downsample=[None, 25, 50, 75],
                train_downsample=[None, 50, 75],
                categorical_perturb_level=["label", "sample"],
                debug=[True],
            )
        )
    )
    N = len(grid)
    tmin, tmax = 2, 10  # seconds
    hmin, hmax = (tmin * N) / 3600, (tmax * N) / 3600
    print(len(grid))
    print(f"Total  1-core CPU runtime estimate: {hmin:0.2f}hrs - {hmax:0.2f}hrs")
    print(
        f"Total 32-core CPU runtime estimate: {hmin / 32:0.2f}hrs - {hmax / 32:0.2f}hrs"
    )
    print(
        f"Total 80-core CPU runtime estimate: {hmin / 80:0.2f}hrs - {hmax / 80:0.2f}hrs"
    )
    return grid


def evaluate(args: dict[str, Any]) -> None:
    try:
        kind: ClassifierKind = args["classifier_kind"]
        hps: Hparams = {
            ClassifierKind.SGD_SVM: SGDLinearSVMHparams().defaults(),
            ClassifierKind.SGD_LR: SGDLRHparams().defaults(),
            ClassifierKind.MLP: MLPHparams().defaults(),
            ClassifierKind.XGBoost: XGBoostHparams().defaults(),
        }[kind]
        evaluator = Evaluator(**args, hparams=hps)
        evaluator.evaluate()
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")


if __name__ == "__main__":
    grid = create_grid()
    process_map(evaluate, grid, total=len(grid), desc="Evaluating", chunksize=1)
