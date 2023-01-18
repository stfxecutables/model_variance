from shutil import rmtree
from time import time
from typing import Any, Callable

import numpy as np
import pytest
from pandas import DataFrame
from pytest import CaptureFixture, raises
from tqdm import tqdm

from src.enumerables import (
    CatPerturbLevel,
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
    RuntimeClass,
)
from src.evaluator import Evaluator
from src.hparams.hparams import Hparams
from src.hparams.logistic import LRHparams, SGDLRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import LinearSVMHparams, SGDLinearSVMHparams, SVMHparams
from src.hparams.xgboost import XGBoostHparams

FASTS = RuntimeClass.very_fasts()
KINDS = [
    ClassifierKind.XGBoost,
    ClassifierKind.SGD_SVM,
    ClassifierKind.SGD_LR,
    ClassifierKind.MLP,
]


def get_evaluator(kind: ClassifierKind, random: bool, i: int) -> Evaluator:
    hp: Hparams = {
        ClassifierKind.XGBoost: XGBoostHparams,
        ClassifierKind.SVM: SVMHparams,
        ClassifierKind.LinearSVM: LinearSVMHparams,
        ClassifierKind.LR: LRHparams,
        ClassifierKind.SGD_LR: SGDLRHparams,
        ClassifierKind.MLP: MLPHparams,
    }[kind]()
    if random:
        hps = hp.random()
    else:
        hps = hp.defaults()
    ds = FASTS[i]
    return Evaluator(
        dataset_name=ds,
        classifier_kind=kind,
        repeat=0,
        run=0,
        dimension_reduction=None,
        continuous_perturb=None,
        categorical_perturb=None,
        hparam_perturb=None,
        train_downsample=None,
        hparams=hps,
        debug=True,
    )


def random_evaluator(_seed: int | None = None) -> Evaluator:
    rng = np.random.default_rng(seed=_seed)

    def choice(x: list[Any]) -> Any:
        val = rng.choice(x)
        if isinstance(val, np.ndarray):
            val = val.item()
        if isinstance(val, np.int64):
            val = int(val)
        return val

    kind = choice(KINDS)
    hp: Hparams = {
        ClassifierKind.XGBoost: XGBoostHparams,
        ClassifierKind.SGD_SVM: SGDLinearSVMHparams,
        ClassifierKind.SGD_LR: SGDLRHparams,
        ClassifierKind.MLP: MLPHparams,
    }[kind]()
    hps = hp.random()
    return Evaluator(
        dataset_name=choice([*DatasetName]),
        classifier_kind=kind,
        repeat=choice(list(range(50))),
        run=choice(list(range(50))),
        dimension_reduction=choice([None, 25, 50, 75]),
        continuous_perturb=choice([*DataPerturbation]),
        categorical_perturb=choice([None, 0.1, 0.2]),
        categorical_perturb_level=choice([*CatPerturbLevel]),
        hparam_perturb=choice([*HparamPerturbation]),
        train_downsample=choice([None, 25, 50, 75]),
        hparams=hps,
        debug=True,
    )


def helper(
    kind: ClassifierKind, random: bool, _capsys: CaptureFixture
) -> list[DataFrame]:
    print("")
    with _capsys.disabled():
        pbar = tqdm(desc=f"{kind.value}: {'':<20}", total=len(FASTS), ncols=80)
    times = []
    for i in range(len(FASTS)):
        start = time()
        try:
            evaluator = get_evaluator(kind=kind, random=random, i=i)
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


def test_ids() -> None:
    for seed in np.random.default_rng().integers(0, int(1e9), 50):
        ev1 = random_evaluator(seed)
        ev2 = random_evaluator(seed)
        ev3 = random_evaluator(seed + 1)
        assert ev1.get_id() == ev2.get_id()
        assert ev1.get_id() != ev3.get_id()


def test_ckpts() -> None:
    hps = SGDLinearSVMHparams().defaults()
    for i in range(3):
        ds = FASTS[i]
        ev_args: dict[str, Any] = dict(
            dataset_name=ds,
            classifier_kind=ClassifierKind.SGD_SVM,
            repeat=0,
            run=0,
            dimension_reduction=None,
            continuous_perturb=None,
            categorical_perturb=None,
            hparam_perturb=None,
            train_downsample=None,
            hparams=hps,
            debug=True,
        )
        ev = Evaluator(**ev_args)
        ckpt_dir = ev.ckpt_file.parent
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if len(ckpts) > 0:
            for ckpt in ckpts:
                ckpt.unlink(missing_ok=True)

        ev.evaluate()
        ev2 = Evaluator(**ev_args)
        with raises(FileExistsError):
            ev2.evaluate()


@pytest.mark.medium
def test_svm_random(capsys: CaptureFixture) -> list[DataFrame]:
    return helper(ClassifierKind.SVM, random=True, _capsys=capsys)


@pytest.mark.medium
def test_linear_svm_random(capsys: CaptureFixture) -> list[DataFrame]:
    return helper(ClassifierKind.LinearSVM, random=True, _capsys=capsys)


@pytest.mark.medium
def test_xgb_random(capsys: CaptureFixture) -> list[DataFrame]:
    return helper(ClassifierKind.XGBoost, random=True, _capsys=capsys)


@pytest.mark.medium
def test_lr_random(capsys: CaptureFixture) -> list[DataFrame]:
    with capsys.disabled():
        return helper(ClassifierKind.LR, random=True, _capsys=capsys)


@pytest.mark.medium
def test_mlp_random(capsys: CaptureFixture) -> list[DataFrame]:
    with capsys.disabled():
        return helper(ClassifierKind.MLP, random=True, _capsys=capsys)


@pytest.mark.medium
def test_svm_default(capsys: CaptureFixture) -> list[DataFrame]:
    return helper(ClassifierKind.SVM, random=False, _capsys=capsys)


@pytest.mark.medium
def test_xgb_default(capsys: CaptureFixture) -> list[DataFrame]:
    return helper(ClassifierKind.XGBoost, random=False, _capsys=capsys)


@pytest.mark.medium
def test_lr_default(capsys: CaptureFixture) -> list[DataFrame]:
    with capsys.disabled():
        return helper(ClassifierKind.LR, random=False, _capsys=capsys)


@pytest.mark.medium
def test_mlp_default(capsys: CaptureFixture) -> list[DataFrame]:
    with capsys.disabled():
        return helper(ClassifierKind.MLP, random=False, _capsys=capsys)
