# from __future__ import annotations

# # fmt: off
# import sys  # isort:skip
# from pathlib import Path  # isort: skip
# ROOT = Path(__file__).resolve().parent.parent  # isort: skip
# sys.path.append(str(ROOT))  # isort: skip
# # fmt: on


from copy import deepcopy
from pathlib import Path
from random import choice
from shutil import rmtree
from typing import Literal, cast

import numpy as np
import pytest
from numpy import ndarray
from sklearn.model_selection import train_test_split

from src.constants import ensure_dir
from src.dataset import Dataset
from src.enumerables import ClassifierKind, DatasetName
from src.evaluator import get_model
from src.hparams.logistic import LRHparams, SGDLRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import LinearSVMHparams, SGDLinearSVMHparams, SVMHparams
from src.hparams.xgboost import XGBoostHparams
from src.models.dl_model import DLModel
from src.models.model import ClassifierModel

Percentage = Literal[None, 25, 50, 75]

ROOT = Path(__file__).resolve().parent.parent  # isort: skip
DIR = ROOT / "__test_temp__"
DIR.mkdir(exist_ok=True)
CATS = [chr(i) for i in (list(range(97, 123)) + list(range(65, 91)))]


def random_classifier(logdir: Path) -> ClassifierModel:
    hp_cls = choice(
        [
            SVMHparams,
            SGDLinearSVMHparams,
            LinearSVMHparams,
            XGBoostHparams,
            SGDLRHparams,
            # These are PyTorch, can't use skops
            # LRHparams,
            # MLPHparams,
        ]
    )
    ds = choice(  # fast
        [
            DatasetName.Anneal,
            DatasetName.Vehicle,
            DatasetName.Arrhythmia,
        ]
    )
    classifier_kind = {
        XGBoostHparams: ClassifierKind.XGBoost,
        SVMHparams: ClassifierKind.SVM,
        SGDLinearSVMHparams: ClassifierKind.SGD_SVM,
        LinearSVMHparams: ClassifierKind.LinearSVM,
        SGDLRHparams: ClassifierKind.SGD_LR,
        # These are PyTorch, can't use skops
        # LRHparams: ClassifierKind.LR,
        # MLPHparams: ClassifierKind.MLP,
    }[hp_cls]
    hps = hp_cls().random()
    return get_model(
        classifier_kind,
        args=dict(hparams=hps, dataset=Dataset(ds), logdir=logdir),
    )


def random_dl_classifier(logdir: Path) -> DLModel:
    hp_cls = choice(
        [
            LRHparams,
            MLPHparams,
        ]
    )
    ds = choice(  # fast
        [
            DatasetName.Anneal,
            DatasetName.Vehicle,
            DatasetName.Arrhythmia,
        ]
    )
    classifier_kind = {
        LRHparams: ClassifierKind.LR,
        MLPHparams: ClassifierKind.MLP,
    }[hp_cls]
    hps = hp_cls().random()
    return cast(
        DLModel,
        get_model(
            classifier_kind,
            args=dict(hparams=hps, dataset=Dataset(ds), logdir=logdir),
        ),
    )


def test_unfitted() -> None:
    for _ in range(10):
        logdir = ensure_dir(DIR / "skops_temp")
        try:
            c1 = random_classifier(logdir)
            c2 = deepcopy(c1)
            with pytest.raises(RuntimeError):
                c1.to_skops(logdir)
            rmtree(logdir)

        except Exception as e:
            if logdir.exists():
                rmtree(logdir)
            raise e
        finally:
            if logdir.exists():
                rmtree(logdir)


def test_fitted() -> None:
    X_tr: ndarray
    X_test: ndarray
    y_tr: ndarray
    y_test: ndarray
    for _ in range(10):
        logdir = ensure_dir(DIR / "skops_temp")
        try:
            c1 = random_classifier(logdir)
            c2 = deepcopy(c1)
            X, y = c1.dataset.get_X_y()
            X_tr, X_test, y_tr, y_test = train_test_split(
                X,
                y,
                test_size=0.5,
                stratify=y,
            )  # type: ignore
            c1.fit(X_tr, y_tr)
            c1.to_skops(logdir)
            pred1 = c1.predict(X_test, y_test)[0]
            c2.model = c1.from_skops(logdir)
            pred2 = c2.predict(X_test, y_test)[0]
            np.testing.assert_array_almost_equal(pred1, pred2)

            rmtree(logdir)

        except Exception as e:
            if logdir.exists():
                rmtree(logdir)
            raise e
        finally:
            if logdir.exists():
                rmtree(logdir)


def test_dl_fitted() -> None:
    X_tr: ndarray
    X_test: ndarray
    y_tr: ndarray
    y_test: ndarray
    for _ in range(10):
        logdir = ensure_dir(DIR / "pytorch_temp")
        try:
            c1 = random_dl_classifier(logdir)
            c2 = deepcopy(c1)
            X, y = c1.dataset.get_X_y()
            X_tr, X_test, y_tr, y_test = train_test_split(
                X,
                y,
                test_size=0.5,
                stratify=y,
            )  # type: ignore
            c1.fit(X_tr, y_tr)

            pred1 = c1.predict(X_test, y_test)[0]
            c2.load_fitted()
            pred2 = c2.predict(X_test, y_test)[0]
            np.testing.assert_array_almost_equal(pred1, pred2)

            rmtree(logdir)

        except Exception as e:
            if logdir.exists():
                rmtree(logdir)
            raise e
        finally:
            if logdir.exists():
                rmtree(logdir)
