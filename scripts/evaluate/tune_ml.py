from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
import os
import sys
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import shuffle
from shutil import copyfile
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from pandas.errors import PerformanceWarning
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from typing_extensions import Literal

from src.constants import BEST_HPS, ensure_dir
from src.enumerables import ClassifierKind, DatasetName, RuntimeClass
from src.evaluator import Tuner, ckpt_file
from src.hparams.hparams import Hparams
from src.hparams.logistic import SGDLRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import NystroemHparams, SGDLinearSVMHparams
from src.hparams.xgboost import XGBoostHparams
from src.parallelize import joblib_map

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
    N_reps = 200
    ss = np.random.SeedSequence(entropy=entropy)
    rngs = [np.random.default_rng(seed) for seed in ss.spawn(N_reps)]

    classifiers = [
        ClassifierKind.SGD_LR,
        ClassifierKind.SGD_SVM,
        ClassifierKind.XGBoost,
        # ClassifierKind.NystroemSVM,  # trash performance
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
            # ClassifierKind.NystroemSVM: NystroemHparams().random(rng=rng),  # trash performance
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
        # print(f"Tuning on data {dsname.name} with classifier {kind.name}")
        tuner.tune()
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")


def get_best_params() -> None:
    TUNING = ROOT / "debug_logs/tuning/debug"
    accfiles = sorted(TUNING.rglob("accuracy.json"))
    tunerfiles = [accfile.parent / "tuner.json" for accfile in accfiles]
    hpfiles = [accfile.parent / "all_hparams.json" for accfile in accfiles]
    all_accs: Dict[str, Dict[str, List[Dict[str, Union[float, Path]]]]] = {}
    for accfile, tunerfile, hpfile in zip(accfiles, tunerfiles, hpfiles):
        tuner = json.loads(tunerfile.read_text())
        dsname = tuner["dataset_name"]
        kind = tuner["classifier_kind"]
        acc = float(accfile.read_text())
        if dsname not in all_accs:
            all_accs[dsname] = {}
        if kind not in all_accs[dsname]:
            all_accs[dsname][kind] = []
        all_accs[dsname][kind].append({"acc": acc, "hpfile": hpfile})

    all_bests: Dict[str, Dict[str, Dict[str, Union[float, Path]]]] = {}
    descs: List[DataFrame] = []
    for dsname, kind_infos in all_accs.items():
        if dsname not in all_bests:
            all_bests[dsname] = {}
        for kind, info in kind_infos.items():
            info = sorted(info, key=lambda d: d["acc"], reverse=True)
            accs = pd.Series([inf["acc"] for inf in info])
            desc = accs.describe(percentiles=[0.01, 0.99]).to_frame().T
            desc["data"] = dsname
            desc["classifier"] = kind
            all_bests[dsname][kind] = info[0]
            descs.append(desc)

    for dsname, kind_info in all_bests.items():
        for kind, info in kind_info.items():
            hpfile = Path(info["hpfile"])  # type: ignore
            outdir = ensure_dir(BEST_HPS / f"{kind}/{dsname}")
            outfile = outdir / "all_hparams.json"
            acc_out = outdir / "accuracy.json"
            copyfile(hpfile, outfile)
            acc_out.write_text(f"{info['acc']:0.20f}")

    desc = pd.concat(descs, axis=0, ignore_index=True)
    print(desc.to_markdown(tablefmt="simple", floatfmt="0.4f"))


if __name__ == "__main__":
    # runtime = RuntimeClass.Fast
    # runtime = RuntimeClass.Mid
    # runtime = RuntimeClass.Slow
    # n_jobs = 40 if runtime is RuntimeClass.Slow else -1
    # dsnames = runtime.members()
    # args = create_args(dsnames=dsnames)
    # joblib_map(evaluate, args, max_workers=n_jobs, desc="Tuning")
    get_best_params()
