from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from math import ceil
from pathlib import Path
from pprint import pformat, pprint
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from scipy.stats import loguniform
from typing_extensions import Literal

from src.enumerables import HparamPerturbation, PerturbMagnitude
from src.perturb import sig_perturb_plus

T = TypeVar("T")


class HparamKind(Enum):
    Continuous = "continuous"
    Ordinal = "ordinal"
    Categorical = "categorical"


class Hparam(ABC, Generic[T]):
    def __init__(self, name: str, value: T | None) -> None:
        super().__init__()
        self.name: str = name
        self._value: T | None = value
        self.kind: HparamKind

    @property
    def value(self) -> T | None:
        return self._value

    @value.setter
    def value(self) -> T:
        raise ValueError("Cannot set value on Hparam")

    @abstractmethod
    def random(self, rng: Generator | None = None) -> Hparam:
        ...

    @abstractmethod
    def perturbed(
        self,
        method: HparamPerturbation,
        magnitude: PerturbMagnitude,
        rng: Generator | None = None,
    ) -> Hparam:
        ...

    @abstractmethod
    def __sub__(self, o: Hparam) -> float:
        if not isinstance(o, Hparam):
            raise ValueError(
                f"Can only find difference between hparams of same kind. Got {type(o)}"
            )
        if o.name != self.name:
            raise ValueError(
                "Cannot find difference between different hparams. "
                f"Got: `{self.name}` - `{o.name}`"
            )
        if self.value is None:
            raise ValueError("Cannot subtract hparam with no value")
        return abs(self.value - o.value)  # type: ignore


class ContinuousHparam(Hparam):
    def __init__(
        self,
        name: str,
        value: float | None,
        max: float,
        min: float,
        log_scale: bool = False,
    ) -> None:
        super().__init__(name=name, value=value)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Continuous
        self._value: float | None = float(value) if value is not None else None
        self.min: float = float(min)
        self.max: float = float(max)
        self.log_scale: bool = log_scale

    def new(self, value: float) -> ContinuousHparam:
        cls: Type[ContinuousHparam] = self.__class__
        return cls(
            name=self.name,
            value=value,
            max=self.max,
            min=self.min,
            log_scale=self.log_scale,
        )

    def random(self, rng: Generator | None = None) -> ContinuousHparam:
        if rng is None:
            if self.log_scale:
                value = float(loguniform.rvs(self.min, self.max, size=1))
            else:
                value = float(np.random.uniform(self.min, self.max))
        else:
            if self.log_scale:
                loguniform.random_state = rng
                value = float(loguniform.rvs(self.min, self.max, size=1))
            else:
                value = float(rng.uniform(self.min, self.max, size=1))
        return self.new(value=value)

    def perturbed(
        self,
        method: HparamPerturbation,
        magnitude: PerturbMagnitude,
        rng: Generator | None = None,
    ) -> Hparam:
        if self.value is None:
            raise ValueError("Cannot perturn hparam if value is None.")
        if rng is None:
            rng = np.random.default_rng()
        mag = magnitude.actual_value()
        if method is HparamPerturbation.SigDig:
            value = float(sig_perturb_plus(self.value, n_digits=mag))
        elif method is HparamPerturbation.RelPercent:
            assert magnitude is PerturbMagnitude.RelPercent10
            if self.log_scale:
                val = np.log10(self.value)
                delta = mag * val  # mag = 0.10
                value = 10 ** rng.uniform(val - delta, val + delta)
            else:
                value = rng.uniform(val - delta, val + delta)
        elif method is HparamPerturbation.AbsPercent:
            assert magnitude is PerturbMagnitude.AbsPercent10
            value = self.val_perturb(rng)
        else:
            raise NotImplementedError()
        val = np.clip(value, a_min=self.min, a_max=self.max)
        return self.new(value=val)

    def val_perturb(self, rng: Generator) -> float:
        if self.log_scale:
            # Example most extreme possible perturbed values:
            #
            #     1.00e-04  ->  [4.47e-05, 2.24e-04]
            #     2.00e-04  ->  [8.93e-05, 4.48e-04]
            #     5.00e-04  ->  [2.23e-04, 1.12e-03]
            val = np.log10(self.value)
            mn, mx = np.log10(self.min), np.log10(self.max)
        else:
            val = self.value
            mn, mx = self.min, self.max

        d = 0.05 * (mx - mn)
        val_min = val - d
        val_max = val + d

        if self.log_scale:
            value = 10 ** rng.uniform(val_min, val_max)
        else:
            value = rng.uniform(val_min, val_max)
        return value

    def normed(self) -> float:
        if self.value is None:
            raise ValueError("Cannot norm if value is undefined.")
        if self.log_scale:
            value = float(np.log10(self.value))
            mn, mx = float(np.log10(self.min)), float(np.log10(self.max))
            return (value - mn) / (mx - mn)
        return float((self.value - self.min) / (self.max - self.min))

    def __sub__(self, o: Hparam) -> float:
        if not isinstance(o, ContinuousHparam):
            raise ValueError(
                f"Can only find difference between hparams of same kind. Got {type(o)}"
            )
        if o.name != self.name:
            raise ValueError(
                "Cannot find difference between different hparams. "
                f"Got: `{self.name}` - `{o.name}`"
            )
        if self.value is None:
            raise ValueError("Cannot subtract hparam with no value")
        return abs(o.normed() - self.normed())

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, value={self.value}, "
            f"min={self.min}, max={self.max}, log={self.log_scale})"
        )

    __repr__ = __str__


class OrdinalHparam(Hparam):
    def __init__(self, name: str, value: int | None, max: int, min: int = 0) -> None:
        super().__init__(name=name, value=value)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Ordinal
        self._value: int | None = int(value) if value is not None else None
        self.min: int = int(min)
        self.max: int = int(max)
        self.values = list(range(self.min, self.max + 1))

    def new(self, value: int) -> OrdinalHparam:
        cls: Type[OrdinalHparam] = self.__class__
        return cls(
            name=self.name,
            value=value,
            max=self.max,
            min=self.min,
        )

    def perturbed(
        self,
        method: HparamPerturbation,
        magnitude: PerturbMagnitude,
        rng: Generator | None = None,
    ) -> Hparam:
        if self.value is None:
            raise ValueError("Cannot perturn hparam if value is None.")
        if rng is None:
            rng = np.random.default_rng()
        value = self.value
        if magnitude not in [
            PerturbMagnitude.SigOne,
            PerturbMagnitude.RelPercent10,
            PerturbMagnitude.AbsPercent10,
        ]:
            raise ValueError("Ordinal perturbation makes sense only for 1 sig dig or 10%")
        mag = magnitude.actual_value()
        if method is HparamPerturbation.SigDig:  # mag == 1
            value = value + rng.integers(-1, 2)
        elif method is HparamPerturbation.RelPercent:  # mag == 0.10
            delta = ceil(mag * value)
            value = rng.integers(value - delta, value + delta + 1)
        elif method is HparamPerturbation.AbsPercent:  # mag == 0.10
            delta = ceil((self.max - self.min) * mag)
            value = value + rng.integers(-delta, delta + 1)
        else:
            raise NotImplementedError()

        val = np.clip(value, a_min=self.min, a_max=self.max)
        return self.new(val)

    def random(self, rng: Generator | None = None) -> OrdinalHparam:
        if rng is None:
            value = int(np.random.choice(self.values))
        else:
            value = int(rng.choice(self.values, size=1, shuffle=False))
        cls: Type[OrdinalHparam] = self.__class__
        return cls(name=self.name, value=value, max=self.max, min=self.min)

    def normed(self) -> float:
        if self.value is None:
            raise ValueError("Cannot norm if value is undefined.")
        return float((self.value - self.min) / (self.max - self.min))

    def __len__(self) -> int:
        return self.max - self.min + 1

    def __sub__(self, o: Hparam) -> float:
        if not isinstance(o, OrdinalHparam):
            raise ValueError(
                f"Can only find difference between hparams of same kind. Got {type(o)}"
            )
        if o.name != self.name:
            raise ValueError(
                "Cannot find difference between different hparams. "
                f"Got: `{self.name}` - `{o.name}`"
            )
        if self.value is None:
            raise ValueError("Cannot subtract hparam with no value")
        return abs(o.normed() - self.normed())

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, value={self.value}, "
            f"min={self.min}, max={self.max})"
        )

    __repr__ = __str__


class CategoricalHparam(Hparam):
    def __init__(
        self, name: str, value: str | None, categories: Sequence[str] | Collection[str]
    ) -> None:
        super().__init__(name=name, value=value)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Categorical
        self._value: str | None = str(value) if value is not None else None
        self.categories: list[str] = sorted(categories)
        self.n_categories: int = len(self.categories)

    def new(self, value: str) -> CategoricalHparam:
        cls: Type[CategoricalHparam] = self.__class__
        return cls(
            name=self.name,
            value=value,
            categories=self.categories,
        )

    def perturbed(
        self,
        method: HparamPerturbation,
        magnitude: PerturbMagnitude = PerturbMagnitude.AbsPercent10,
        rng: Generator | None = None,
    ) -> Hparam:
        if self.value is None:
            raise ValueError("Cannot perturb categorical hparam with value None.")
        if method is HparamPerturbation.SigDig:
            # No coherent definition for this, also sig dig perturbation
            # is supposed to be "invisible", so perhaps leaving no impact on
            # categoricals makes most sense here.
            return self.new(self.value)
        mag = 0.10
        rng = np.random.default_rng() if rng is None else rng
        if rng.uniform(0, 1) < mag:
            return self.random(rng)
        return self.new(self.value)

    def random(self, rng: Generator | None = None) -> CategoricalHparam:
        if rng is None:
            value = str(np.random.choice(self.categories, size=1))
        else:
            value = str(rng.choice(self.categories, size=1, shuffle=False))
        return self.new(value)

    def __len__(self) -> int:
        return self.n_categories

    def __sub__(self, o: Hparam) -> float:
        if not isinstance(o, CategoricalHparam):
            raise ValueError(
                f"Can only find difference between hparams of same kind. Got {type(o)}"
            )
        if o.name != self.name:
            raise ValueError(
                "Cannot find difference between different hparams. "
                f"Got: `{self.name}` - `{o.name}`"
            )
        if self.value is None:
            raise ValueError("Cannot subtract hparam with no value")
        return float(int(self.value != o.value))

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, value={self.value}, "
            f"categories={self.categories})"
        )

    __repr__ = __str__


class Hparams(ABC):
    def __init__(
        self, hparams: Collection[Hparam] | Sequence[Hparam] | None = None
    ) -> None:
        super().__init__()
        hps = sorted(hparams, key=lambda hp: hp.name) if hparams is not None else []
        self.hparams: dict[str, Hparam] = {}
        for hp in hps:
            if hp.name in self.hparams:
                raise ValueError(f"Hparam with duplicate name {hp.name} found.")
            self.hparams[hp.name] = hp

        self.n_hparams = self.N = len(self.hparams)
        self.continuous: dict[str, Hparam] = {}
        self.ordinals: dict[str, Hparam] = {}
        self.categoricals: dict[str, Hparam] = {}
        for name, hp in self.hparams.items():
            if hp.kind is HparamKind.Continuous:
                self.continuous[name] = hp
            elif hp.kind is HparamKind.Ordinal:
                self.ordinals[name] = hp
            elif hp.kind is HparamKind.Categorical:
                self.categoricals[name] = hp
            else:
                raise ValueError("Invalid Hparam kind!")
        self.n_continuous = len(self.continuous)
        self.n_ordinal = len(self.ordinals)
        self.n_categorical = len(self.categoricals)

    def random(self, rng: Generator | None = None) -> Hparams:
        cls = self.__class__
        hps = []
        for name, hp in self.hparams.items():
            hps.append(hp.random(rng))
        return cls(hparams=hps)

    def __sub__(self, o: Hparams) -> float:
        if not isinstance(o, Hparams):
            raise ValueError(f"Can only find difference between instance of Hparams.")
        if sorted(self.hparams.keys()) != sorted(o.hparams.keys()):
            raise ValueError(
                "Cannot find difference between different sets of hparams. "
                f"Got left: {self.hparams} and right: {o.hparams}"
            )
        diffs = []
        for name in sorted(self.hparams.keys()):
            hp1 = self.hparams[name]
            hp2 = o.hparams[name]
            diffs.append(hp1 - hp2)
        return float(np.mean(diffs))

    def __str__(self) -> str:
        fmt = [f"{self.__class__.__name__}("]
        for name, hp in self.hparams.items():
            fmt.append(f"  {str(hp)}")
        fmt.append(")")
        return "\n".join(fmt)
