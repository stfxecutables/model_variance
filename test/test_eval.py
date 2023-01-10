import random
from argparse import Namespace
from shutil import rmtree

import numpy as np
import pytest
from pytest import CaptureFixture
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.dataset import Dataset
from src.enumerables import ClassifierKind, DataPerturbation, DatasetName, RuntimeClass
from src.evaluator import Evaluator
from src.hparams.logistic import LRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import SVMHparams
from src.hparams.xgboost import XGBoostHparams

FASTS = RuntimeClass.very_fasts()


def get_evaluator(kind: ClassifierKind, i: int) -> Evaluator:
    hps = {
        ClassifierKind.XGBoost: XGBoostHparams,
        ClassifierKind.SVM: SVMHparams,
        ClassifierKind.LR: LRHparams,
        ClassifierKind.MLP: MLPHparams,
    }[kind]().random()
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


def helper(kind: ClassifierKind, _capsys: CaptureFixture) -> None:
    print("")
    with _capsys.disabled():
        pbar = tqdm(desc=f"{kind.value}: {'':<20}", total=len(FASTS), ncols=80)
    for i in range(len(FASTS)):
        try:
            evaluator = get_evaluator(kind=kind, i=i)
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
            if evaluator.logdir.exists():
                rmtree(evaluator.logdir)
            raise e
    pbar.close()


@pytest.mark.medium
def test_svm(capsys: CaptureFixture) -> None:
    helper(ClassifierKind.SVM, capsys)


@pytest.mark.medium
def test_xgb(capsys: CaptureFixture) -> None:
    helper(ClassifierKind.XGBoost, capsys)


@pytest.mark.medium
def test_lr(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        helper(ClassifierKind.LR, capsys)


@pytest.mark.medium
def test_mlp(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        helper(ClassifierKind.MLP, capsys)
