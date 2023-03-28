from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from warnings import filterwarnings

from pandas.errors import PerformanceWarning
from sklearn.model_selection import ParameterGrid

from src.enumerables import (
    CatPerturbLevel,
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
    RuntimeClass,
)
from src.evaluator import Evaluator, ckpt_file
from src.hparams.hparams import Hparams
from src.hparams.light import LightGBMHparams
from src.hparams.logistic import SGDLRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import SGDLinearSVMHparams
from src.hparams.xgboost import XGBoostHparams
from src.parallelize import joblib_map

filterwarnings("ignore", category=PerformanceWarning)

"""
Need to rethink this a bit. Using the full data makes
"""


def create_grid(dsnames: Optional[List[DatasetName]] = None) -> List[Dict[str, Any]]:
    classifiers = [
        ClassifierKind.SGD_LR,
        ClassifierKind.SGD_SVM,
        ClassifierKind.XGBoost,
        ClassifierKind.LightGBM,
    ]
    data_perturbs = [
        None,
        DataPerturbation.DoubleNeighbor,
        DataPerturbation.FullNeighbor,
        DataPerturbation.RelPercent20,
        DataPerturbation.Percentile20,
        DataPerturbation.SigDigZero,
    ]
    hp_perturbs = [
        None,
        HparamPerturbation.SigZero,
        HparamPerturbation.RelPercent20,
        HparamPerturbation.AbsPercent20,
    ]
    dsnames = RuntimeClass.most_fastest() if dsnames is None else dsnames
    grid = list(
        ParameterGrid(
            dict(
                dataset_name=dsnames,
                classifier_kind=classifiers,
                repeat=list(range(10)),
                run=list(range(10)),
                # hparams needs to be handled manually
                dimension_reduction=["cat"],
                continuous_perturb=data_perturbs,
                # categorical_perturb=[None, 0.1],
                categorical_perturb=[None],
                hparam_perturb=hp_perturbs,
                # train_downsample=[None, 25, 50, 75],
                train_downsample=[None, 50, 75],
                categorical_perturb_level=[CatPerturbLevel.Sample],
                label=["prelim"],
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


def evaluate(args: Dict[str, Any]) -> None:
    try:
        ckpt_args = {**args}
        ckpt_args.pop("debug")
        ckpt_args["mode"] = "debug"
        ckpt = ckpt_file(**ckpt_args)
        if ckpt.exists():
            return
        dsname: DatasetName = args["dataset_name"]
        kind: ClassifierKind = args["classifier_kind"]
        hps: Hparams = {
            ClassifierKind.SGD_SVM: lambda: SGDLinearSVMHparams.tuned(dsname),
            ClassifierKind.SGD_LR: lambda: SGDLRHparams.tuned(dsname),
            ClassifierKind.MLP: lambda: MLPHparams.tuned(dsname),
            ClassifierKind.XGBoost: lambda: XGBoostHparams.tuned(dsname),
            ClassifierKind.LightGBM: lambda: LightGBMHparams.tuned(dsname),
        }[kind]()
        evaluator = Evaluator(**args, base_hps=hps)
        evaluator.evaluate()
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")


if __name__ == "__main__":
    # 21 600 runs about an hour for Anneal
    grid = create_grid(
        dsnames=[
            # DatasetName.Anneal,
            DatasetName.Arrhythmia,
            DatasetName.Christine,
            DatasetName.Credit_g,
            DatasetName.Vehicle,
        ]
    )
    cluster = os.environ.get("CC_CLUSTER")
    if cluster == "niagara":
        max_workers = 80
    elif cluster == "cedar":
        max_workers = 32
    else:
        max_workers = 8
    joblib_map(evaluate, grid, desc="Evaluating", max_workers=max_workers)
