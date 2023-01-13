import traceback
from dataclasses import dataclass
from random import choice
from shutil import rmtree
from time import time
from typing import Any

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
from src.hparams.svm import LinearSVMHparams, SGDLinearSVMHparams, SVMHparams
from src.hparams.xgboost import XGBoostHparams

# rename prevents recursive pytest discovery

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
    pd.options.display.width = 500


def get_evaluator(targs: TimingArgs) -> Evaluator:
    kind = targs.kind
    dsname = targs.dsname

    hp: Hparams = {
        ClassifierKind.XGBoost: XGBoostHparams,
        ClassifierKind.SGD_SVM: SGDLinearSVMHparams,
        ClassifierKind.LinearSVM: LinearSVMHparams,
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
        {"elapsed_s": float("nan"), "dataset": targs.dsname.name, "acc": -1.0},
        index=[0],
    )
    try:
        start = time()
        evaluator = get_evaluator(targs)

        acc = evaluator.evaluate(return_test_acc=True)
        elapsed = time() - start
        duration = DataFrame(
            {"elapsed_s": elapsed, "dataset": targs.dsname.name, "acc": acc},
            index=[0],
        )
        if evaluator.logdir.exists():
            rmtree(evaluator.logdir)
    except Exception:
        if evaluator.logdir.exists():  # type: ignore
            rmtree(evaluator.logdir)  # type: ignore
        traceback.print_exc()
        return duration
    return duration


def get_times_sequential(
    kind: ClassifierKind, runtime: RuntimeClass, repeats: int, _capsys: CaptureFixture
) -> DataFrame:
    print("")

    dsnames = runtime.members()
    targs: list[TimingArgs] = []
    for _ in range(repeats):
        targs.extend([TimingArgs(kind=kind, dsname=name) for name in dsnames])
    dfs = []
    with _capsys.disabled():
        fmt = f"{kind.name} - {{}}"
        pbar = tqdm(total=len(targs), desc=f"Fitting {runtime.value} runtime models")
        for targ in targs:
            pbar.set_description(fmt.format(targ.dsname.name))
            df = get_time(targ)
            dfs.append(df)
            pbar.update()
        pbar.close()
    return pd.concat(dfs, axis=0, ignore_index=True)


def get_times_parallel(
    kind: ClassifierKind, runtime: RuntimeClass, repeats: int, _capsys: CaptureFixture
) -> DataFrame:
    print("")

    dsnames = runtime.members()
    targs = []
    for _ in range(repeats):
        targs.extend([TimingArgs(kind=kind, dsname=name) for name in dsnames])
    with _capsys.disabled():
        times = process_map(
            get_time,
            targs,
            total=len(targs),
            desc=f"Fitting {runtime.value} models",
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
    kind: ClassifierKind,
    runtime: RuntimeClass,
    repeats: int,
    parallel: bool,
    _capsys: CaptureFixture,
) -> None:
    with _capsys.disabled():
        args: dict[str, Any] = dict(
            kind=kind, runtime=runtime, repeats=repeats, _capsys=_capsys
        )
        if parallel:
            df = get_times_parallel(**args)
        else:
            df = get_times_sequential(**args)

    outfile = {
        RuntimeClass.Fast: FAST_RUNTIMES,
        RuntimeClass.Mid: MED_RUNTIMES,
        RuntimeClass.Slow: SLOW_RUNTIMES,
    }[runtime] / f"{kind.value}_{runtime.value}_runtimes.json"
    df.to_json(outfile)
    with _capsys.disabled():
        set_long_print()
        df_orig = df.copy()
        df = (
            df_orig.groupby("dataset")
            .describe()
            .sort_values(by=("elapsed_s", "max"), ascending=False)
        )
        runtimes = (
            df["elapsed_s"]  # type:ignore
            .drop(columns="count")
            .loc[:, ["min", "mean", "50%", "max", "std"]]
            .rename(columns={"50%": "med"})
            .applymap(to_readable)
        )
        accs = (
            df["acc"]  # type:ignore
            .sort_values(by="max", ascending=False)
            .drop(columns="count")
            .loc[:, ["min", "mean", "50%", "max"]]
            .rename(columns={"50%": "med"})
            .rename(columns=lambda s: f"acc_{s}")
        )
        accs["acc_range"] = accs["acc_max"] - accs["acc_min"]
        accs = accs.round(4)
        info = pd.concat([runtimes, accs], axis=1)
        print(info)
        print(f"Saved all {runtime.value} runtimes to {outfile}")
        summary_out = outfile.parent / f"{outfile.stem}_summary.json"
        info.to_json(summary_out)
        print(f"Saved {runtime.value} runtime summaries to {summary_out}")


# def test_svm_fast(capsys: CaptureFixture) -> None:
#     summarize_times(
#         kind=ClassifierKind.SVM,
#         runtime=RuntimeClass.Fast,
#         repeats=5,
#         parallel=True,
#         _capsys=capsys,
#     )


class TestFast:
    def test_sgd_svm(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.SGD_SVM,
            runtime=RuntimeClass.Fast,
            repeats=5,
            parallel=True,
            _capsys=capsys,
        )

    def test_linear_svm(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.LinearSVM,
            runtime=RuntimeClass.Fast,
            repeats=5,
            parallel=True,
            _capsys=capsys,
        )

    def test_xgb(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.XGBoost,
            runtime=RuntimeClass.Fast,
            repeats=5,
            parallel=True,
            _capsys=capsys,
        )

    def test_lr(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.LR,
            runtime=RuntimeClass.Fast,
            repeats=5,
            parallel=False,
            _capsys=capsys,
        )

    def test_sgd_lr(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.SGD_LR,
            runtime=RuntimeClass.Fast,
            repeats=5,
            parallel=False,
            _capsys=capsys,
        )

    def test_mlp(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.MLP,
            runtime=RuntimeClass.Fast,
            repeats=1,
            parallel=False,
            _capsys=capsys,
        )


# mediums


# def test_svm_med(capsys: CaptureFixture) -> None:
#     summarize_times(
#         kind=ClassifierKind.SVM,
#         runtime=RuntimeClass.Mid,
#         repeats=5,
#         parallel=False,
#         _capsys=capsys,
#     )


class TestMed:
    def test_sgd_svm(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.SGD_SVM,
            runtime=RuntimeClass.Mid,
            repeats=5,
            parallel=False,
            _capsys=capsys,
        )

    def test_xgb(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.XGBoost,
            runtime=RuntimeClass.Mid,
            repeats=5,
            parallel=False,
            _capsys=capsys,
        )

    def test_lr(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.LR,
            runtime=RuntimeClass.Mid,
            repeats=5,
            parallel=False,
            _capsys=capsys,
        )

    def test_sgd_lr(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.SGD_LR,
            runtime=RuntimeClass.Mid,
            repeats=5,
            parallel=False,
            _capsys=capsys,
        )

    def test_mlp(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.MLP,
            runtime=RuntimeClass.Mid,
            repeats=1,
            parallel=False,
            _capsys=capsys,
        )

    # def test_linear_svm(self, capsys: CaptureFixture) -> None:
    #     summarize_times(
    #         kind=ClassifierKind.LinearSVM,
    #         runtime=RuntimeClass.Mid,
    #         repeats=5,
    #         parallel=True,
    #         _capsys=capsys,
    #     )


class TestSlow:
    def test_sgd_svm(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.SGD_SVM,
            runtime=RuntimeClass.Slow,
            repeats=5,
            parallel=True,
            _capsys=capsys,
        )

    def test_xgb(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.XGBoost,
            runtime=RuntimeClass.Slow,
            repeats=5,
            parallel=False,
            _capsys=capsys,
        )

    def test_lr(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.LR,
            runtime=RuntimeClass.Slow,
            repeats=5,
            parallel=False,
            _capsys=capsys,
        )

    def test_sgd_lr(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.SGD_LR,
            runtime=RuntimeClass.Slow,
            repeats=5,
            parallel=False,
            _capsys=capsys,
        )

    def test_mlp(self, capsys: CaptureFixture) -> None:
        summarize_times(
            kind=ClassifierKind.MLP,
            runtime=RuntimeClass.Slow,
            repeats=3,
            parallel=False,
            _capsys=capsys,
        )
