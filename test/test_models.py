from math import ceil
from pathlib import Path
from random import choice
from shutil import rmtree
from tempfile import mkdtemp
from uuid import uuid4

import numpy as np

from src.dataset import Dataset
from src.enumerables import ClassifierKind, DataPerturbation, DatasetName, RuntimeClass
from src.evaluator import Evaluator
from src.hparams.hparams import (
    CategoricalHparam,
    ContinuousHparam,
    Hparam,
    HparamPerturbation,
    Hparams,
    OrdinalHparam,
)
from src.hparams.svm import SVMHparams
from src.hparams.xgboost import XGBoostHparams
from src.utils import missing_keys
from test.helpers import (
    random_categorical,
    random_continuous,
    random_hparams,
    random_ordinal,
)
from src.models.logistic import LRModel

ROOT = Path(__file__).resolve().parent.parent  # isort: skip
DIR = ROOT / "__test_temp__"
DIR.mkdir(exist_ok=True)


def test_lr() -> None:
    for dsname in RuntimeClass.Fast.members()[:2]:
        ds = Dataset(dsname)
        model = LRModel()
        X_train, y_train, X_test, y_test = ds.get_monte_carlo_splits()
