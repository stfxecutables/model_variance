from shutil import rmtree

import pytest
from pytest import CaptureFixture
from tqdm import tqdm

from src.enumerables import ClassifierKind, RuntimeClass
from src.evaluator import Evaluator
from src.hparams.hparams import Hparams
from src.hparams.logistic import LRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import SVMHparams
from src.hparams.xgboost import XGBoostHparams

FASTS = RuntimeClass.very_fasts()


def get_evaluator(kind: ClassifierKind, random: bool, i: int) -> Evaluator:
    hp: Hparams = {
        ClassifierKind.XGBoost: XGBoostHparams,
        ClassifierKind.SVM: SVMHparams,
        ClassifierKind.LR: LRHparams,
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


def helper(kind: ClassifierKind, random: bool, _capsys: CaptureFixture) -> None:
    print("")
    with _capsys.disabled():
        pbar = tqdm(desc=f"{kind.value}: {'':<20}", total=len(FASTS), ncols=80)
    for i in range(len(FASTS)):
        try:
            evaluator = get_evaluator(kind=kind, random=random, i=i)
            with _capsys.disabled():
                pbar.set_description(f"{kind.value}: {evaluator.dataset_name.name:<20}")
            evaluator.evaluate(no_pred=False)
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


@pytest.mark.medium
def test_svm_random(capsys: CaptureFixture) -> None:
    helper(ClassifierKind.SVM, random=True, _capsys=capsys)


@pytest.mark.medium
def test_xgb_random(capsys: CaptureFixture) -> None:
    helper(ClassifierKind.XGBoost, random=True, _capsys=capsys)


@pytest.mark.medium
def test_lr_random(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        helper(ClassifierKind.LR, random=True, _capsys=capsys)


@pytest.mark.medium
def test_mlp_random(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        helper(ClassifierKind.MLP, random=True, _capsys=capsys)


@pytest.mark.medium
def test_svm_default(capsys: CaptureFixture) -> None:
    helper(ClassifierKind.SVM, random=False, _capsys=capsys)


@pytest.mark.medium
def test_xgb_default(capsys: CaptureFixture) -> None:
    helper(ClassifierKind.XGBoost, random=False, _capsys=capsys)


@pytest.mark.medium
def test_lr_default(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        helper(ClassifierKind.LR, random=False, _capsys=capsys)


@pytest.mark.medium
def test_mlp_default(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        helper(ClassifierKind.MLP, random=False, _capsys=capsys)
