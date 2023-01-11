from random import choice
from shutil import rmtree
from time import time

import pandas as pd
from pandas import DataFrame
from pytest import CaptureFixture
from tqdm import tqdm

from src.constants import RESULTS, ensure_dir
from src.enumerables import ClassifierKind, RuntimeClass
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


def unfuck_pandas_printing() -> None:
    pd.options.display.max_rows = 5000
    pd.options.display.max_info_rows = 5000
    pd.options.display.max_columns = 1000
    pd.options.display.max_info_columns = 1000
    pd.options.display.large_repr = "truncate"
    pd.options.display.expand_frame_repr = True
    pd.options.display.width = 200


def get_evaluator(kind: ClassifierKind, runtime: RuntimeClass, i: int) -> Evaluator:
    hp: Hparams = {
        ClassifierKind.XGBoost: XGBoostHparams,
        ClassifierKind.SVM: SVMHparams,
        ClassifierKind.LR: LRHparams,
        ClassifierKind.MLP: MLPHparams,
    }[kind]()
    hps = hp.random()
    ds = runtime.members()[i]
    return Evaluator(
        dataset_name=ds,
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


def helper(
    kind: ClassifierKind, runtime: RuntimeClass, _capsys: CaptureFixture
) -> list[DataFrame]:
    print("")
    with _capsys.disabled():
        pbar = tqdm(desc=f"{kind.value}: {'':<20}", total=len(FASTS), ncols=80)
    times = []
    for i in range(len(runtime.members())):
        start = time()
        try:
            evaluator = get_evaluator(kind=kind, runtime=runtime, i=i)
            with _capsys.disabled():
                pbar.set_description(f"{kind.value}: {evaluator.dataset_name.name:<20}")
            evaluator.evaluate(no_pred=False)
            elapsed = time() - start
            times.append(
                DataFrame(
                    {"elapsed_s": elapsed, "dataset": evaluator.dataset_name.name},
                    index=[0],
                )
            )
            assert (evaluator.preds_dir / "preds.npz").exists()
            if evaluator.logdir.exists():
                rmtree(evaluator.logdir)
            with _capsys.disabled():
                pbar.update()
        except Exception as e:
            pbar.close()
            if evaluator.logdir.exists():  # type: ignore
                rmtree(evaluator.logdir)  # type: ignore
            raise e
    pbar.close()
    return times


def test_svm_fast(capsys: CaptureFixture) -> None:
    times = []
    for r in range(2):
        with capsys.disabled():
            print(f"Repeat {r}:")
        runtime = helper(ClassifierKind.SVM, runtime=RuntimeClass.Fast, _capsys=capsys)
        times.extend(runtime)

    df = pd.concat(times, axis=0, ignore_index=True)
    with capsys.disabled():
        unfuck_pandas_printing()
        runtimes = (
            df.groupby("dataset")
            .describe()["elapsed_s"]  # type:ignore
            .sort_values(by="max", ascending=False)
        )
        print(runtimes)
