from pathlib import Path
from random import choice
from shutil import rmtree
from tempfile import mkdtemp
from uuid import uuid4

import numpy as np

from src.enumerables import ClassifierKind, DataPerturbation, DatasetName
from src.evaluator import Evaluator
from src.hparams.hparams import (
    CategoricalHparam,
    ContinuousHparam,
    Hparam,
    Hparams,
    OrdinalHparam,
)
from src.hparams.svm import SVMHparams
from src.hparams.xgboost import XGBoostHparams
from src.utils import missing_keys

ROOT = Path(__file__).resolve().parent.parent  # isort: skip
DIR = ROOT / "__test_temp__"
DIR.mkdir(exist_ok=True)
CATS = [chr(i) for i in (list(range(97, 123)) + list(range(65, 91)))]
