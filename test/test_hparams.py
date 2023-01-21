from math import ceil
from pathlib import Path

import numpy as np
from pytest import raises

from src.hparams.hparams import Hparam, HparamPerturbation
from src.hparams.logistic import SGDLRHparams
from src.hparams.mlp import MLPHparams
from src.hparams.svm import SGDLinearSVMHparams
from src.hparams.xgboost import XGBoostHparams
from test.helpers import (
    random_categorical,
    random_continuous,
    random_fixed,
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
        for method in [*HparamPerturbation]:
            equals = []
            hp = random_categorical()
            for _ in range(N):
                h2 = hp.perturbed(method=method)
                equals.append(int(hp - h2) == 0)
            total = sum(equals)
            if method in [HparamPerturbation.SigOne, HparamPerturbation.SigZero]:
                assert total == N
            else:
                # preturbation is random, but the probability all are equal approaches
                # nil as the number of perturbed copies grows large
                assert (total > 0) and (total < N), f"Problem for method: {method}"


class TestFixed:
    def test_clone(self) -> None:
        for _ in range(50):
            hp = random_fixed()
            h2 = hp.clone()
            assert hp == h2

    def test_perturb(self) -> None:
        N = 50
        for method in [*HparamPerturbation]:
            hp = random_fixed()
            for _ in range(N):
                h2 = hp.perturbed(method=method)
                assert h2 == hp

    def test_subtract(self) -> None:
        N = 10
        for _ in range(N):
            hp = random_fixed()
            assert (hp - hp.clone()) == 0.0
        for _ in range(N):
            h1 = random_fixed()
            h2 = random_fixed()
            with raises(ValueError):
                h1 - h2  # type: ignore
        for _ in range(N):
            h1 = random_fixed()
            h2 = h1.clone()
            h2._value = h1._value + 1
            with raises(RuntimeError):
                h1 - h2  # type: ignore


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

    def test_perturb_rel(self) -> None:
        for method in [
            HparamPerturbation.RelPercent05,
            HparamPerturbation.RelPercent10,
            HparamPerturbation.RelPercent20,
        ]:
            mag = float(method.magnitude())
            for _ in range(self.N):
                hp = random_ordinal()
                assert hp.value is not None
                delta = ceil(mag * hp.value)
                h2 = hp.perturbed(method=method)
                assert (hp - h2) <= delta

    def test_perturb_abs(self) -> None:
        for method in [
            HparamPerturbation.AbsPercent05,
            HparamPerturbation.AbsPercent10,
            HparamPerturbation.AbsPercent20,
        ]:
            mag = method.magnitude()
            for _ in range(self.N):
                hp = random_ordinal()
                delta = ceil((hp.max - hp.min) * mag)
                h2 = hp.perturbed(method=method)
                assert (hp - h2) <= delta


class TestContinuous:
    N = 250
    ALLOWANCE = N // 20
    # N = 250

    def test_clone(self) -> None:
        for _ in range(50):
            hp = random_continuous()
            h2 = hp.clone()
            assert hp == h2

    def test_perturb_sig_one(self) -> None:
        method = HparamPerturbation.SigOne
        equals = []
        for _ in range(self.N):
            hp = random_continuous()
            h2 = hp.perturbed(method=method)
            equals.append(hp == h2)
        assert sum(equals) <= self.ALLOWANCE

    def test_perturb_rel(self) -> None:
        for method in [
            HparamPerturbation.RelPercent05,
            HparamPerturbation.RelPercent10,
            HparamPerturbation.RelPercent20,
        ]:
            mag = method.magnitude()
            equals = []
            for _ in range(self.N):
                hp = random_continuous()
                # delta = ceil(mag * hp.value)
                h2 = hp.perturbed(method=method)
                equals.append(hp == h2)
                assert hp.value is not None
                assert h2.value is not None
                if hp.log_scale:
                    diff = abs(np.log10(hp.value) - np.log10(h2.value))
                    assert diff <= abs(mag * np.log10(hp.value))
                else:
                    diff = hp - h2
                    assert diff <= mag * hp.value
            assert sum(equals) <= self.ALLOWANCE

    def test_perturb_abs(self) -> None:
        for method in [
            HparamPerturbation.AbsPercent05,
            HparamPerturbation.AbsPercent10,
            HparamPerturbation.AbsPercent20,
        ]:
            mag = method.magnitude()
            equals = []
            for _ in range(self.N):
                hp = random_continuous()
                # delta = ceil((hp.max - hp.min) * mag)
                h2 = hp.perturbed(method=method)
                equals.append(hp == h2)
                diff = hp - h2
                assert diff <= mag * abs(hp.max - hp.min)
            assert sum(equals) <= self.ALLOWANCE


class TestDefaults:
    def test_sgd_svm(self) -> None:
        hps = SGDLinearSVMHparams().defaults()
        hp: Hparam
        for name, hp in hps.hparams.items():
            if hp.value is None:
                raise ValueError(f"Got None in hyperparameter '{name}': {hp}")

    def test_sgd_lr(self) -> None:
        hps = SGDLRHparams().defaults()
        hp: Hparam
        for name, hp in hps.hparams.items():
            if hp.value is None:
                raise ValueError(f"Got None in hyperparameter '{name}': {hp}")

    def test_xgboost(self) -> None:
        hps = XGBoostHparams().defaults()
        hp: Hparam
        for name, hp in hps.hparams.items():
            if hp.value is None:
                raise ValueError(f"Got None in hyperparameter '{name}': {hp}")

    def test_mlp(self) -> None:
        hps = MLPHparams().defaults()
        hp: Hparam
        for name, hp in hps.hparams.items():
            if hp.value is None:
                raise ValueError(f"Got None in hyperparameter '{name}': {hp}")
