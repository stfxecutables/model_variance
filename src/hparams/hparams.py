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
from pathlib import Path
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
from pandas import DataFrame, Series
from typing_extensions import Literal

T = TypeVar("T")


class HparamKind(Enum):
    Continuous = "continuous"
    Ordinal = "ordinal"
    Categorical = "categorical"


class Hparam(ABC, Generic[T]):
    def __init__(self, name: str, value: T) -> None:
        super().__init__()
        self.name: str = name
        self._value: T = value
        self.kind: HparamKind

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self) -> T:
        raise ValueError("Cannot set value on Hparam")

    @abstractmethod
    def __sub__(self, o: Hparam) -> int:
        if not isinstance(o, Hparam):
            raise ValueError(
                f"Can only find difference between hparams of same kind. Got {type(o)}"
            )
        if o.name != self.name:
            raise ValueError(
                "Cannot find difference between different hparams. "
                f"Got: `{self.name}` - `{o.name}`"
            )
        return abs(self.value - o.value)


class ContinuousHparam(Hparam):
    def __init__(
        self, name: str, value: float, max: float, min: float, log_scale: bool = False
    ) -> None:
        super().__init__(name=name, value=value)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Continuous
        self._value: float = float(value)
        self.min: float = float(min)
        self.max: float = float(max)
        self.log_scale: bool = log_scale

    def normed(self) -> float:
        return (self.value - self.min) / (self.max - self.min)

    def __sub__(self, o: ContinuousHparam) -> float:
        if not isinstance(o, ContinuousHparam):
            raise ValueError(
                f"Can only find difference between hparams of same kind. Got {type(o)}"
            )
        if o.name != self.name:
            raise ValueError(
                "Cannot find difference between different hparams. "
                f"Got: `{self.name}` - `{o.name}`"
            )
        return abs(o.normed() - self.normed())


class OrdinalHparam(Hparam):
    def __init__(self, name: str, value: int, max: int, min: int = 0) -> None:
        super().__init__(name=name, value=value)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Ordinal
        self._value: int = int(value)
        self.min: int = int(min)
        self.max: int = int(max)

    def normed(self) -> float:
        return (self.value - self.min) / (self.max - self.min)

    def __len__(self) -> int:
        return self.max - self.min + 1

    def __sub__(self, o: OrdinalHparam) -> int:
        if not isinstance(o, OrdinalHparam):
            raise ValueError(
                f"Can only find difference between hparams of same kind. Got {type(o)}"
            )
        if o.name != self.name:
            raise ValueError(
                "Cannot find difference between different hparams. "
                f"Got: `{self.name}` - `{o.name}`"
            )
        return abs(o.normed() - self.normed())


class CategoricalHparam(Hparam):
    def __init__(self, name: str, value: str | None, n_categories: int) -> None:
        super().__init__(name=name, value=value)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Categorical
        self._value: str | None = value
        self.n_categories: int = int(n_categories)

    def __len__(self) -> int:
        return self.n_categories

    def __sub__(self, o: CategoricalHparam) -> int:
        if not isinstance(o, CategoricalHparam):
            raise ValueError(
                f"Can only find difference between hparams of same kind. Got {type(o)}"
            )
        if o.name != self.name:
            raise ValueError(
                "Cannot find difference between different hparams. "
                f"Got: `{self.name}` - `{o.name}`"
            )
        return int(self.value != o.value)


class Hparams(ABC):
    def __init__(self, hparams: Collection[Hparam] | Sequence[Hparam]) -> None:
        super().__init__()
        hps = sorted(hparams, key=lambda hp: hp.name)
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
            diffs.append(float(hp1 - hp2))
        return float(np.mean(diffs))
