import traceback
from argparse import Namespace
from dataclasses import dataclass
from random import choice
from shutil import rmtree
from time import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from pytest import CaptureFixture
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.constants import RESULTS, ensure_dir
from src.enumerables import ClassifierKind, DatasetName, RuntimeClass
from src.evaluator import Evaluator
from src.hparams.hparams import Hparams
from src.hparams.logistic import LRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import SVMHparams
from src.hparams.xgboost import XGBoostHparams

# rename prevents recursive pytest discovery
from test.test_eval import test_lr_random as lr_random
from test.test_eval import test_mlp_random as mlp_random
from test.test_eval import test_svm_random as svm_random
from test.test_eval import test_xgb_random as xgb_random

RUNTIMES = ensure_dir(RESULTS / "runtimes")
FAST_RUNTIMES = ensure_dir(RUNTIMES / "fast")
MED_RUNTIMES = ensure_dir(RUNTIMES / "med")
SLOW_RUNTIMES = ensure_dir(RUNTIMES / "slow")

FASTS = RuntimeClass.Fast.members()
MEDS = RuntimeClass.Mid.members()
SLOWS = RuntimeClass.Slow.members()


@dataclass
class TimingArgs:
    kind: ClassifierKind
    dsname: DatasetName


def set_long_print() -> None:
    pd.options.display.max_rows = 5000
    pd.options.display.max_info_rows = 5000
    pd.options.display.max_columns = 1000
    pd.options.display.max_info_columns = 1000
    pd.options.display.large_repr = "truncate"
    pd.options.display.expand_frame_repr = True
    pd.options.display.width = 200


def get_evaluator(targs: TimingArgs) -> Evaluator:
    kind = targs.kind
    dsname = targs.dsname

    hp: Hparams = {
        ClassifierKind.XGBoost: XGBoostHparams,
        ClassifierKind.SVM: SVMHparams,
        ClassifierKind.LR: LRHparams,
        ClassifierKind.MLP: MLPHparams,
    }[kind]()
    hps = hp.random()
    return Evaluator(
        dataset_name=dsname,
        classifier_kind=kind,
        repeat=choice(range(50)),
        run=choice(range(50)),
        dimension_reduction=None,
        continuous_perturb=None,
        categorical_perturb=None,
        hparam_perturb=None,
        train_downsample=None,
        hparams=hps,
        debug=True,
    )


def get_time(targs: TimingArgs) -> DataFrame:
    duration = DataFrame(
        {"elapsed_s": float("nan"), "dataset": targs.dsname.name},
        index=[0],
    )
    try:
        start = time()
        evaluator = get_evaluator(targs)

        evaluator.evaluate(no_pred=False)
        elapsed = time() - start
        duration = DataFrame(
            {"elapsed_s": elapsed, "dataset": targs.dsname.name},
            index=[0],
        )
        assert (evaluator.preds_dir / "preds.npz").exists()
        if evaluator.logdir.exists():
            rmtree(evaluator.logdir)
    except Exception:
        if evaluator.logdir.exists():  # type: ignore
            rmtree(evaluator.logdir)  # type: ignore
        traceback.print_exc()
        return duration
    return duration


def get_times(
    kind: ClassifierKind, runtime: RuntimeClass, repeats: int, _capsys: CaptureFixture
) -> DataFrame:
    print("")

    dsnames = runtime.members()
    targs = []
    for _ in range(repeats):
        targs.extend([TimingArgs(kind=kind, dsname=name) for name in dsnames])
    args = dict(max_workers=1) if kind is ClassifierKind.MLP else dict()
    with _capsys.disabled():
        times = process_map(
            get_time,
            targs,
            total=len(targs),
            desc=f"Fitting {runtime.value} models",
            **args,
        )
    return pd.concat(times, axis=0, ignore_index=True)


def to_readable(duration_s: float) -> str:
    if duration_s <= 120:
        return f"{np.round(duration_s, 1):03.1f} sec"
    mins = duration_s / 60
    if mins <= 120:
        return f"{np.round(mins, 1):03.1f} min"
    hrs = mins / 60
    return f"{np.round(hrs, 2):03.2f} hrs"


def summarize_times(
    kind: ClassifierKind, runtime: RuntimeClass, repeats: int, _capsys: CaptureFixture
) -> None:
    with _capsys.disabled():
        df = get_times(kind=kind, runtime=runtime, repeats=repeats, _capsys=_capsys)

    outfile = {
        RuntimeClass.Fast: FAST_RUNTIMES,
        RuntimeClass.Mid: MED_RUNTIMES,
        RuntimeClass.Slow: SLOW_RUNTIMES,
    }[runtime] / f"{kind.value}_{runtime.value}_runtimes.json"
    df.to_json(outfile)
    with _capsys.disabled():
        set_long_print()
        runtimes = (
            df.groupby("dataset")
            .describe()["elapsed_s"]  # type:ignore
            .sort_values(by="max", ascending=False)
            .drop(columns="count")
            .applymap(to_readable)
        )
        print(runtimes)
        print(f"Saved all {runtime.value} runtimes to {outfile}")
        summary_out = outfile.parent / f"{outfile.stem}_summary.json"
        runtimes.to_json(summary_out)
        print(f"Saved {runtime.value} runtime summaries to {summary_out}")


def test_svm_fast(capsys: CaptureFixture) -> None:
    summarize_times(
        kind=ClassifierKind.SVM, runtime=RuntimeClass.Fast, repeats=5, _capsys=capsys
    )


def test_xgb_fast(capsys: CaptureFixture) -> None:
    summarize_times(
        kind=ClassifierKind.XGBoost, runtime=RuntimeClass.Fast, repeats=5, _capsys=capsys
    )


def test_lr_fast(capsys: CaptureFixture) -> None:
    summarize_times(
        kind=ClassifierKind.LR, runtime=RuntimeClass.Fast, repeats=5, _capsys=capsys
    )


def test_mlp_fast(capsys: CaptureFixture) -> None:
    summarize_times(
        kind=ClassifierKind.MLP, runtime=RuntimeClass.Fast, repeats=1, _capsys=capsys
    )


# mediums

def test_svm_med(capsys: CaptureFixture) -> None:
    summarize_times(
        kind=ClassifierKind.SVM, runtime=RuntimeClass.Mid, repeats=5, _capsys=capsys
    )


def test_xgb_med(capsys: CaptureFixture) -> None:
    summarize_times(
        kind=ClassifierKind.XGBoost, runtime=RuntimeClass.Mid, repeats=5, _capsys=capsys
    )


def test_lr_med(capsys: CaptureFixture) -> None:
    summarize_times(
        kind=ClassifierKind.LR, runtime=RuntimeClass.Mid, repeats=5, _capsys=capsys
    )


def test_mlp_med(capsys: CaptureFixture) -> None:
    summarize_times(
        kind=ClassifierKind.MLP, runtime=RuntimeClass.Mid, repeats=1, _capsys=capsys
    )
