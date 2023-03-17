from itertools import combinations
from shutil import rmtree
from typing import Any, Dict

import numpy as np
import pytest

from src.enumerables import ClassifierKind, DatasetName, RuntimeClass
from src.evaluator import Evaluator, Tuner
from src.hparams.hparams import Hparams
from src.hparams.logistic import SGDLRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import SGDLinearSVMHparams
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
        dimension_reduction="cat",
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
        ev_args: Dict[str, Any] = dict(
            dataset_name=ds,
            classifier_kind=ClassifierKind.SGD_SVM,
            repeat=0,
            run=0,
            hparams=hps,
            dimension_reduction="cat",
            debug=True,
        )
        tuner = Tuner(**ev_args)
        ckpt_dir = tuner.ckpt_file.parent
        try:

            res = tuner.tune(skip_done=True)
            assert tuner.acc_file.exists()
            assert tuner.ckpt_file.exists()

            tuner2 = Tuner(**ev_args)
            res2 = tuner2.tune(skip_done=True)
            assert not tuner2.logdir.exists()
            np.testing.assert_approx_equal(res, res2)
        except Exception as e:
            raise e
        finally:
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if len(ckpts) > 0:
                for ckpt in ckpts:
                    ckpt.unlink(missing_ok=True)


HPS = [
    XGBoostHparams,
    SGDLRHparams,
    SGDLinearSVMHparams,
    MLPHparams,
]
CASES = [(dsname, hp) for dsname in DatasetName for hp in HPS]


@pytest.mark.parametrize("dsname,hps", CASES)
def test_load_tuned_hps(dsname: DatasetName, hps: Hparams) -> None:
    hps.tuned(dsname)
