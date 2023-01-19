from pathlib import Path
from shutil import rmtree
from tarfile import TarFile
from tarfile import open as tar_open
from time import time
from typing import Any

import numpy as np
import pytest
from pandas import DataFrame
from pytest import CaptureFixture
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


def random_evaluator(rng: np.random.Generator | None = None) -> Evaluator:
    if rng is None:
        rng = np.random.default_rng()

    def choice(x: list[Any]) -> Any:
        val = rng.choice(x)
        if isinstance(val, np.ndarray):
            val = val.item()
        if isinstance(val, np.int64):  # type: ignore
            val = int(val)
        return val

    kind = choice(KINDS)
    hp: Hparams = {
        ClassifierKind.XGBoost: XGBoostHparams,
        ClassifierKind.SGD_SVM: SGDLinearSVMHparams,
        ClassifierKind.SGD_LR: SGDLRHparams,
        ClassifierKind.MLP: MLPHparams,
    }[kind]()
    hps = hp.random(rng)
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
    # extremely unlikely to get replicates, but technically can fail randomly
    seqs = np.random.SeedSequence().spawn(51)
    for i in range(50):
        rng1 = np.random.default_rng(seqs[i])
        rng2 = np.random.default_rng(seqs[i])
        rng3 = np.random.default_rng(seqs[i + 1])
        ev1 = random_evaluator(rng1)
        ev2 = random_evaluator(rng2)
        ev3 = random_evaluator(rng3)
        try:
            assert ev1.get_id() == ev2.get_id()
            assert ev1 == ev2

            assert ev1.get_id() != ev3.get_id()
            assert ev1 != ev3
        except AssertionError as e:
            raise e
        finally:
            ev1.cleanup()
            ev2.cleanup()
            ev3.cleanup()


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
        ev2 = Evaluator(**ev_args)
        try:
            assert not ev.ckpt_file.exists()
            ev.evaluate(skip_done=False)
            assert ev.ckpt_file.exists()

            assert ev2.model.fitted is False
            ev2.evaluate(skip_done=True)
            assert ev2.model.fitted is False
        except Exception as e:
            raise e
        finally:
            ev.cleanup()
            ev2.cleanup()


def test_archival() -> None:
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
        tf = ev.logdir.parent / f"{ev.logdir.name}.tar.gz"
        try:
            ev.evaluate(skip_done=False)
            assert tf.exists()
            tar: TarFile
            with tar_open(tf, "r:gz") as tar:
                names = [Path(name).name for name in tar.getnames()]
                assert "preds.npz" in names
                assert "targs.npz" in names
                assert "evaluator.json" in names
                assert "hparams" in names

        except Exception as e:
            raise e
        finally:
            ev.cleanup()
            if tf.exists():
                tf.unlink()


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
