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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
from warnings import filterwarnings

import numpy as np
from numpy.random import Generator
from pandas.errors import PerformanceWarning
from sklearn.model_selection import ParameterGrid
from tqdm.contrib.concurrent import process_map

from src.enumerables import (
    CatPerturbLevel,
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
    RuntimeClass,
)
from src.evaluator import Tuner, ckpt_file
from src.hparams.hparams import Hparams
from src.hparams.logistic import SGDLRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import SGDLinearSVMHparams
from src.hparams.xgboost import XGBoostHparams

filterwarnings("ignore", category=PerformanceWarning)


@dataclass
class ParallelArgs:
    dataset_name: DatasetName
    classifier_kind: ClassifierKind
    repeat: int
    rng: Generator


def create_args(dsnames: Optional[List[DatasetName]] = None) -> List[ParallelArgs]:
    # from a random SeedSequence init
    entropy = 298356256812349700484752502215766469641
    N_reps = 100
    ss = np.random.SeedSequence(entropy=entropy)
    rngs = [np.random.default_rng(seed) for seed in ss.spawn(N_reps)]

    classifiers = [
        ClassifierKind.SGD_LR,
        ClassifierKind.SGD_SVM,
        ClassifierKind.XGBoost,
    ]
    dsnames = RuntimeClass.most_fastest() if dsnames is None else dsnames
    grid = [
        Namespace(**args)
        for args in list(
            ParameterGrid(
                dict(
                    dataset_name=dsnames,
                    classifier_kind=classifiers,
                    repeat=list(range(N_reps)),
                    # run=[0],
                    # dimension_reduction=["cat"],
                    # debug=[True],
                )
            )
        )
    ]
    args = [
        ParallelArgs(
            dataset_name=args.dataset_name,
            classifier_kind=args.classifier_kind,
            repeat=args.repeat,
            rng=rngs[args.repeat],
        )
        for args in grid
    ]
    return args


def evaluate(args: ParallelArgs) -> None:
    try:
        dsname = args.dataset_name
        kind = args.classifier_kind
        repeat = args.repeat
        run = 0
        dimension_reduction = "cat"
        continuous_perturb = None
        categorical_perturb = None
        hparam_perturb = None
        train_downsample = None
        tune = True

        ckpt = ckpt_file(
            dataset_name=dsname,
            classifier_kind=kind,
            repeat=repeat,
            run=run,
            dimension_reduction=dimension_reduction,
            continuous_perturb=continuous_perturb,
            categorical_perturb=categorical_perturb,
            hparam_perturb=hparam_perturb,
            train_downsample=train_downsample,
            tune=tune,
        )
        if ckpt.exists():
            return
        rng = args.rng
        hps: Hparams = {
            ClassifierKind.SGD_SVM: SGDLinearSVMHparams().random(rng=rng),
            ClassifierKind.SGD_LR: SGDLRHparams().random(rng=rng),
            ClassifierKind.MLP: MLPHparams().random(rng=rng),
            ClassifierKind.XGBoost: XGBoostHparams().random(rng=rng),
        }[kind]
        tuner = Tuner(
            dataset_name=dsname,
            classifier_kind=kind,
            repeat=repeat,
            run=run,
            hparams=hps,
            dimension_reduction=dimension_reduction,
            debug=True,
        )
        tuner.tune()
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")


if __name__ == "__main__":
    # 21 600 runs about an hour for Anneal
    # dsnames = RuntimeClass.very_fasts()
    dsnames = RuntimeClass.Fast.members()
    args = create_args(dsnames=dsnames)
    process_map(
        evaluate, args, total=len(args), desc="Tuning", chunksize=1, smoothing=0.08
    )
