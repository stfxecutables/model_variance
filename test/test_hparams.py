from math import ceil
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

ROOT = Path(__file__).resolve().parent.parent  # isort: skip
DIR = ROOT / "__test_temp__"
DIR.mkdir(exist_ok=True)
CATS = [chr(i) for i in (list(range(97, 123)) + list(range(65, 91)))]


class TestCategorical:
    def test_clone(self) -> None:
        for _ in range(50):
            hp = random_categorical()
            h2 = hp.clone()
            assert hp == h2

    def test_perturb(self) -> None:
        N = 250
        for method in [
            HparamPerturbation.SigOne,
            HparamPerturbation.RelPercent10,
            HparamPerturbation.AbsPercent10,
        ]:
            equals = []
            hp = random_categorical()
            for _ in range(N):
                h2 = hp.perturbed(method=method)
                equals.append(int(hp - h2) == 0)
            total = sum(equals)
            if method is HparamPerturbation.SigOne:
                assert total == N
            else:
                # preturbation is random, but the probability all are equal approaches
                # nil as the number of perturbed copies grows large
                assert (total > 0) and (total < N)


class TestOrdinal:
    N = 250

    def test_clone(self) -> None:
        for _ in range(50):
            hp = random_ordinal()
            h2 = hp.clone()
            assert hp == h2

    def test_perturb_sig_one(self) -> None:
        method = HparamPerturbation.SigOne
        for _ in range(self.N):
            hp = random_ordinal()
            h2 = hp.perturbed(method=method)
            assert (hp - h2) < 2

    def test_perturb_rel_10(self) -> None:
        method = HparamPerturbation.RelPercent10
        mag = method.magnitude()
        for _ in range(self.N):
            hp = random_ordinal()
            delta = ceil(mag * hp.value)
            h2 = hp.perturbed(method=method)
            assert (hp - h2) <= delta

    def test_perturb_abs_10(self) -> None:
        method = HparamPerturbation.AbsPercent10
        mag = method.magnitude()
        for _ in range(self.N):
            hp = random_ordinal()
            delta = ceil((hp.max - hp.min) * mag)
            h2 = hp.perturbed(method=method)
            assert (hp - h2) <= delta
