import traceback
from shutil import rmtree
from time import time
from typing import Any

import pytest
from pandas import DataFrame
from pytest import CaptureFixture, raises
from tqdm import tqdm

from src.enumerables import ClassifierKind, DatasetName, RuntimeClass
from src.evaluator import Evaluator, Tuner
from src.hparams.hparams import Hparams
from src.hparams.logistic import LRHparams, SGDLRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import LinearSVMHparams, SGDLinearSVMHparams, SVMHparams
from src.hparams.xgboost import XGBoostHparams

FASTS = RuntimeClass.very_fasts()


def get_tuner(kind: ClassifierKind, random: bool, i: int) -> Evaluator:
    hp: Hparams = {
        ClassifierKind.XGBoost: XGBoostHparams,
        ClassifierKind.SGD_LR: SGDLRHparams,
        ClassifierKind.SGD_SVM: SGDLinearSVMHparams,
        ClassifierKind.MLP: MLPHparams,
    }[kind]()
    if random:
        hps = hp.random()
    else:
        hps = hp.defaults()
    ds = FASTS[i]
    return Tuner(
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


def test_save_dirs() -> None:
    for kind in [
        ClassifierKind.XGBoost,
        ClassifierKind.SGD_LR,
        ClassifierKind.SGD_SVM,
        ClassifierKind.MLP,
    ]:
        try:
            tuner = get_tuner(kind, random=False, i=0)
            assert "tuning" in str(tuner.setup_logdir())
            assert "tuning" in str(tuner.logdir)
            assert "tuning" in str(tuner.ckpt_file)
            assert "tuning" in str(tuner.res_dir)
        except AssertionError as e:
            raise e
        finally:
            tuner = get_tuner(kind, random=False, i=0)
            logdir = tuner.setup_logdir()
            if logdir.exists():
                rmtree(logdir)


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
        ev = Tuner(**ev_args)
        ckpt_dir = ev.ckpt_file.parent
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if len(ckpts) > 0:
            for ckpt in ckpts:
                ckpt.unlink(missing_ok=True)

        ev.evaluate()
        ev2 = Tuner(**ev_args)
        with raises(FileExistsError):
            ev2.evaluate()
