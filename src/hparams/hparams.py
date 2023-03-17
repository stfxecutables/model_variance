from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
from abc import abstractmethod
from argparse import Namespace
from enum import Enum
from math import ceil
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from numpy.random import Generator
from scipy.stats import loguniform
from scipy.stats.qmc import Halton

from src.constants import BEST_HPS
from src.enumerables import ClassifierKind, DatasetName, HparamPerturbation
from src.perturb import sig_perturb_plus
from src.serialize import DirJSONable, FileJSONable

T = TypeVar("T", covariant=True)
H = TypeVar("H", bound="Hparam", covariant=True)


class HparamKind(Enum):
    Continuous = "continuous"
    Ordinal = "ordinal"
    Categorical = "categorical"
    Fixed = "fixed"


class Hparam(FileJSONable["Hparam"], Generic[T, H]):
    def __init__(
        self, name: str, value: Optional[T], default: Optional[T] = None
    ) -> None:
        super().__init__()
        self.name: str = name
        self._value: Optional[T] = value
        self._default: Optional[T] = default
        self.kind: HparamKind

    def to_dict(self) -> Dict[str, Optional[T]]:
        return {self.name: self.value}

    @abstractmethod
    def to_json_dict(self) -> Dict[str, Any]:
        ...

    def to_jsons(self) -> str:
        return json.dumps(
            self.to_json_dict(),
            indent=2,
        )

    @property
    def value(self) -> Optional[T]:
        return self._value

    @value.setter
    def value(self) -> T:
        raise ValueError("Cannot set value on Hparam")

    def default(self) -> Hparam:
        return self.new(value=self._default)

    @abstractmethod
    def random(self, rng: Optional[Generator] = None) -> Hparam:
        ...

    @abstractmethod
    def perturbed(
        self,
        method: HparamPerturbation,
        rng: Optional[Generator] = None,
    ) -> Hparam:
        ...

    @abstractmethod
    def new(self, value: Optional[T]) -> Hparam:
        ...

    @abstractmethod
    def clone(self) -> Hparam:
        ...

    @staticmethod
    def from_dict(hpdict: Dict[str, T]) -> Hparam:
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hparam):
            raise TypeError(f"Cannot compare types {type(self)} and {type(other)}")

        return (
            (self.name == other.name)
            and (self.value == other.value)
            and (self.kind is other.kind)
        )


class ContinuousHparam(Hparam):
    def __init__(
        self,
        name: str,
        value: Optional[float],
        max: float,
        min: float,
        log_scale: bool = False,
        default: Optional[float] = None,
    ) -> None:
        super().__init__(name=name, value=value, default=default)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Continuous
        self._value: Optional[float] = float(value) if value is not None else None
        self.min: float = float(min)
        self.max: float = float(max)
        self.log_scale: bool = log_scale

        if self.min <= 0 and self.log_scale:
            raise RuntimeError("Cannot have a log-scale hparam and min=0")

    def new(self, value: float) -> ContinuousHparam:
        cls: Type[ContinuousHparam] = self.__class__
        return cls(
            name=self.name,
            value=value,
            max=self.max,
            min=self.min,
            log_scale=self.log_scale,
        )

    def clone(self) -> Hparam:
        cls: Type[ContinuousHparam] = self.__class__
        return cls(
            name=self.name,
            value=self.value,
            max=self.max,
            min=self.min,
            log_scale=self.log_scale,
        )

    def random(self, rng: Optional[Generator] = None) -> ContinuousHparam:
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
        rng: Optional[Generator] = None,
    ) -> Hparam:
        if self.value is None:
            raise ValueError("Cannot perturb hparam if value is None.")
        if rng is None:
            rng = np.random.default_rng()
        mag = method.magnitude()
        if method in [HparamPerturbation.SigOne, HparamPerturbation.SigZero]:
            value = float(sig_perturb_plus(self.value, n_digits=int(mag)))
        elif method in [
            HparamPerturbation.RelPercent05,
            HparamPerturbation.RelPercent10,
            HparamPerturbation.RelPercent20,
        ]:
            val = np.log10(self.value) if self.log_scale else self.value
            delta = mag * val
            vmin = val - delta
            vmax = val + delta
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            raw = rng.uniform(vmin, vmax)
            value = 10**raw if self.log_scale else raw
        elif method in [
            HparamPerturbation.AbsPercent05,
            HparamPerturbation.AbsPercent10,
            HparamPerturbation.AbsPercent20,
        ]:
            value = self.val_perturb(mag, rng)
        else:
            raise NotImplementedError(
                f"Continuous perturbation not implemented for {method.name}"
            )
        val = np.clip(value, a_min=self.min, a_max=self.max)
        return self.new(value=val)

    def val_perturb(self, mag: float, rng: Generator) -> float:
        if self.value is None:
            raise ValueError("Cannot perturn hparam if value is None.")
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

        d = (mag / 2) * (mx - mn)
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

    def to_json(self, path: Path) -> None:
        with open(path, "w") as handle:
            json.dump(
                {
                    "name": self.name,
                    "value": self.value,
                    "default": self._default,
                    "min": self.min,
                    "max": self.max,
                    "log_scale": self.log_scale,
                    "kind": self.kind.value,
                },
                handle,
                indent=2,
            )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "default": self._default,
            "min": self.min,
            "max": self.max,
            "log_scale": self.log_scale,
            "kind": self.kind.value,
        }

    @staticmethod
    def from_json(path: Path) -> ContinuousHparam:
        with open(path, "r") as handle:
            d = Namespace(**json.load(handle))
        return ContinuousHparam(
            name=d.name,
            value=d.value,
            min=d.min,
            max=d.max,
            log_scale=d.log_scale,
            default=d.default,
        )

    @staticmethod
    def from_dict(hpdict: Dict[str, Any]) -> ContinuousHparam:
        d = Namespace(**hpdict)
        return ContinuousHparam(
            name=d.name,
            value=d.value,
            min=d.min,
            max=d.max,
            log_scale=d.log_scale,
            default=d.default,
        )

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
    def __init__(
        self,
        name: str,
        value: Optional[int],
        max: int,
        min: int = 0,
        default: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, value=value, default=default)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Ordinal
        self._value: Optional[int] = int(value) if value is not None else None
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

    def clone(self) -> Hparam:
        cls: Type[OrdinalHparam] = self.__class__
        return cls(
            name=self.name,
            value=self.value,
            max=self.max,
            min=self.min,
        )

    def perturbed(
        self,
        method: HparamPerturbation,
        rng: Optional[Generator] = None,
    ) -> Hparam:
        if self.value is None:
            raise ValueError("Cannot perturn hparam if value is None.")

        if rng is None:
            rng = np.random.default_rng()
        value = self.value
        if method not in [
            HparamPerturbation.SigOne,
            HparamPerturbation.SigZero,
            HparamPerturbation.RelPercent05,
            HparamPerturbation.RelPercent10,
            HparamPerturbation.RelPercent20,
            HparamPerturbation.AbsPercent05,
            HparamPerturbation.AbsPercent10,
            HparamPerturbation.AbsPercent20,
        ]:
            raise ValueError(f"Got invalid perturbation method: {method}")
        mag = method.magnitude()
        if method in [HparamPerturbation.SigOne, HparamPerturbation.SigZero]:
            # treat all as +-1
            value = value + rng.integers(-1, 2)
        elif method in [
            HparamPerturbation.RelPercent05,
            HparamPerturbation.RelPercent10,
            HparamPerturbation.RelPercent20,
        ]:
            delta = ceil(mag * value)
            value = rng.integers(value - delta, value + delta + 1)
        elif method in [
            HparamPerturbation.AbsPercent05,
            HparamPerturbation.AbsPercent10,
            HparamPerturbation.AbsPercent20,
        ]:
            delta = ceil((self.max - self.min) * (mag / 2))
            value = value + rng.integers(-delta, delta + 1)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

        val = np.clip(value, a_min=self.min, a_max=self.max)
        return self.new(val)

    def random(self, rng: Optional[Generator] = None) -> OrdinalHparam:
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

    def to_json(self, path: Path) -> None:
        with open(path, "w") as handle:
            json.dump(
                {
                    "name": self.name,
                    "value": self.value,
                    "default": self._default,
                    "min": self.min,
                    "max": self.max,
                    "kind": self.kind.value,
                },
                handle,
                indent=2,
            )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "default": self._default,
            "min": self.min,
            "max": self.max,
            "kind": self.kind.value,
        }

    @staticmethod
    def from_json(path: Path) -> OrdinalHparam:
        with open(path, "r") as handle:
            d = Namespace(**json.load(handle))
        return OrdinalHparam(
            name=d.name,
            value=d.value,
            default=d.default,
            min=d.min,
            max=d.max,
        )

    @staticmethod
    def from_dict(hpdict: Dict[str, Any]) -> OrdinalHparam:
        d = Namespace(**hpdict)
        return OrdinalHparam(
            name=d.name,
            value=d.value,
            default=d.default,
            min=d.min,
            max=d.max,
        )

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
        self,
        name: str,
        value: Optional[Any],
        categories: Union[Sequence[Any], Collection[Any]],
        default: Optional[Any] = None,
    ) -> None:
        super().__init__(name=name, value=value, default=default)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Categorical
        self._value: Optional[Any] = value if value is not None else None
        self.categories: List[Any] = sorted(categories, key=str)
        self.n_categories: int = len(self.categories)

    def new(self, value: Any) -> CategoricalHparam:
        cls: Type[CategoricalHparam] = self.__class__
        return cls(
            name=self.name,
            value=value,
            categories=self.categories,
        )

    def clone(self) -> Hparam:
        cls: Type[CategoricalHparam] = self.__class__
        return cls(
            name=self.name,
            value=self.value,
            categories=self.categories,
        )

    def perturbed(
        self,
        method: HparamPerturbation = HparamPerturbation.AbsPercent10,
        rng: Optional[Generator] = None,
    ) -> Hparam:
        if self.value is None:
            raise ValueError("Cannot perturb categorical hparam with value None.")
        if method in [HparamPerturbation.SigOne, HparamPerturbation.SigZero]:
            # No coherent definition for this, also sig dig perturbation
            # is supposed to be "invisible", so perhaps leaving no impact on
            # categoricals makes most sense here.
            return self.new(self.value)
        mag = method.magnitude()
        rng = np.random.default_rng() if rng is None else rng
        if rng.uniform(0, 1) < mag:
            return self.random(rng)
        return self.new(self.value)

    def random(self, rng: Optional[Generator] = None) -> CategoricalHparam:
        if rng is None:
            value = np.random.choice(self.categories, size=1).item()
        else:
            value = rng.choice(self.categories, size=1, shuffle=False).item()
        return self.new(value)

    def to_json(self, path: Path) -> None:
        with open(path, "w") as handle:
            json.dump(
                {
                    "name": self.name,
                    "value": self.value,
                    "default": self._default,
                    "categories": self.categories,
                    "kind": self.kind.value,
                },
                handle,
                indent=2,
            )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "default": self._default,
            "categories": self.categories,
            "kind": self.kind.value,
        }

    @staticmethod
    def from_json(path: Path) -> CategoricalHparam:
        with open(path, "r") as handle:
            d = Namespace(**json.load(handle))
        return CategoricalHparam(
            name=d.name,
            value=d.value,
            default=d.default,
            categories=d.categories,
        )

    @staticmethod
    def from_dict(hpdict: Dict[str, Any]) -> CategoricalHparam:
        d = Namespace(**hpdict)
        return CategoricalHparam(
            name=d.name,
            value=d.value,
            default=d.default,
            categories=d.categories,
        )

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


class FixedHparam(Hparam, Generic[T]):
    def __init__(self, name: str, value: T, default: Optional[Any] = None) -> None:
        super().__init__(name=name, value=value, default=default)
        self.name: str = name
        self.kind: HparamKind = HparamKind.Fixed
        self._value: T = value
        self._default: T = value

    def new(self, value: Optional[T]) -> FixedHparam:
        cls: Type[FixedHparam] = self.__class__
        return cls(
            name=self.name,
            value=self.value,
        )

    def clone(self) -> Hparam:
        cls: Type[FixedHparam] = self.__class__
        return cls(
            name=self.name,
            value=self.value,
        )

    def perturbed(
        self,
        method: HparamPerturbation = HparamPerturbation.AbsPercent10,
        rng: Optional[Generator] = None,
    ) -> Hparam:
        return self.new(self.value)

    def random(self, rng: Optional[Generator] = None) -> FixedHparam:
        return self.new(self.value)

    def to_json(self, path: Path) -> None:
        with open(path, "w") as handle:
            json.dump(
                {
                    "name": self.name,
                    "value": self.value,
                    "default": self._default,
                    "kind": self.kind.value,
                },
                handle,
                indent=2,
            )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "default": self._default,
            "kind": self.kind.value,
        }

    @staticmethod
    def from_json(path: Path) -> FixedHparam:
        with open(path, "r") as handle:
            d = Namespace(**json.load(handle))
        return FixedHparam(
            name=d.name,
            value=d.value,
            default=d.default,
        )

    @staticmethod
    def from_dict(hpdict: Dict[str, Any]) -> FixedHparam:
        d = Namespace(**hpdict)
        return FixedHparam(
            name=d.name,
            value=d.value,
            default=d.default,
        )

    def __sub__(self, o: Hparam) -> float:
        if not isinstance(o, FixedHparam):
            raise ValueError(
                f"Can only find difference between hparams of same kind. Got {type(o)}"
            )
        if o.name != self.name:
            raise ValueError(
                "Cannot find difference between different hparams. "
                f"Got: `{self.name}` - `{o.name}`"
            )
        if self.value != o.value:
            raise RuntimeError(
                "Trying to compare FixedHparams with same name and different values. "
                "This should not be possible, and suggests a de-serialization or "
                "other issue."
            )
        return 0.0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, value={self.value}, "

    __repr__ = __str__


class Hparams(DirJSONable):
    kind: ClassifierKind

    def __init__(
        self, hparams: Optional[Union[Collection[Hparam], Sequence[Hparam]]] = None
    ) -> None:
        super().__init__()
        hps = sorted(hparams, key=lambda hp: hp.name) if hparams is not None else []
        self.hparams: Dict[str, Hparam] = {}
        for hp in hps:
            if hp.name in self.hparams:
                raise ValueError(f"Hparam with duplicate name {hp.name} found.")
            self.hparams[hp.name] = hp

        self.n_hparams = self.N = len(self.hparams)
        self.continuous: Dict[str, ContinuousHparam] = {}
        self.ordinals: Dict[str, OrdinalHparam] = {}
        self.categoricals: Dict[str, CategoricalHparam] = {}
        self.fixeds: Dict[str, FixedHparam] = {}
        for name, hp in self.hparams.items():
            if hp.kind is HparamKind.Continuous:
                self.continuous[name] = hp  # type: ignore
            elif hp.kind is HparamKind.Ordinal:
                self.ordinals[name] = hp  # type: ignore
            elif hp.kind is HparamKind.Categorical:
                self.categoricals[name] = hp  # type: ignore
            elif hp.kind is HparamKind.Fixed:
                self.fixeds[name] = hp  # type: ignore
            else:
                raise ValueError("Invalid Hparam kind!")
        self.n_continuous = len(self.continuous)
        self.n_ordinal = len(self.ordinals)
        self.n_categorical = len(self.categoricals)
        self.n_fixed = len(self.fixeds)

    def to_dict(self) -> Dict[str, Union[int, float, str]]:
        d = {}
        for hp in self.hparams.values():
            if hp.name in d:
                raise KeyError(
                    f"Duplicate key: {hp.name}. This means two hparams have same name."
                )
            d.update(hp.to_dict())
        return d

    def clone(self) -> Hparams:
        cls: Type[Hparams] = self.__class__
        clones = [hp.clone() for hp in self.hparams.values()]
        return cls(clones)

    def defaults(self) -> Hparams:
        cls: Type[Hparams] = self.__class__
        defaulteds = [hp.default() for hp in self.hparams.values()]
        return cls(defaulteds)

    @classmethod
    def tuned(cls, dsname: DatasetName) -> Hparams:
        file = BEST_HPS / f"{cls.kind.value}/{dsname.value}/all_hparams.json"
        if not file.exists():
            raise FileNotFoundError(
                f"Tuned hparams for {cls.kind.name} on data {dsname.name} not found."
            )
        return cls.from_json(file.parent)

    def perturbed(
        self,
        method: HparamPerturbation = HparamPerturbation.AbsPercent10,
        rng: Optional[Generator] = None,
    ) -> Hparams:
        cls: Type[Hparams] = self.__class__
        perturbs = [hp.perturbed(method=method, rng=rng) for hp in self.hparams.values()]
        return cls(perturbs)

    def random(self, rng: Optional[Generator] = None) -> Hparams:
        cls = self.__class__
        hps = []
        for name, hp in self.hparams.items():
            hps.append(hp.random(rng))
        return cls(hparams=hps)

    def quasirandom(
        self, iteration: Optional[int] = None, rng: Optional[Generator] = None
    ) -> Hparams:
        cls = self.__class__
        # not sure we can do this with one generator unfortunately
        Halton_cnt = Halton(d=self.n_continuous, seed=rng)
        Halton_ord = Halton(d=self.n_ordinal + self.n_categorical, seed=rng)
        if (iteration is not None) and (iteration > 0):
            Halton_cnt.fast_forward(iteration)
            Halton_ord.fast_forward(iteration)

        hps: List[Hparam] = []
        conts = Halton_cnt.random()
        hpc: ContinuousHparam
        for i, (name, hpc) in enumerate(self.continuous.items()):
            cont = conts[0][i]
            hmin, hmax = hpc.min, hpc.max
            if hpc.log_scale:
                hmin, hmax = np.log10(hmin), np.log10(hmax)
            hval = hmin + (hmax - hmin) * cont
            if hpc.log_scale:
                hval = 10**hval
            hps.append(hpc.new(hval))

        if Halton_ord.d > 0:
            hpo: OrdinalHparam
            hpt: CategoricalHparam
            l_bounds = [hpo.min for hpo in self.ordinals.values()] + [
                0 for hpt in self.categoricals.values()
            ]
            u_bounds = [hpo.max + 1 for hpo in self.ordinals.values()] + [
                hpt.n_categories + 1 for hpt in self.categoricals.values()
            ]
            ords_cats = Halton_ord.integers(l_bounds=l_bounds, u_bounds=u_bounds, n=1)

            for i, (name, hpo) in enumerate(self.ordinals.items()):
                hval = ords_cats[0][i]
                hps.append(hpo.new(hval))

            for i, (name, hpt) in enumerate(self.categoricals.items(), self.n_ordinal):
                h_idx = ords_cats[0][i]
                hval = hpt.categories[h_idx]
                hps.append(hpt.new(hval))

        hps.extend([hpf.clone() for hpf in self.fixeds.values()])
        return cls(hparams=hps)

    def to_json(self, root: Path) -> None:
        root.mkdir(exist_ok=True, parents=True)
        outfile = root / "all_hparams.json"
        inners = []
        hparam: Hparam
        for name, hparam in self.hparams.items():
            inners.append(hparam.to_json_dict())
        outfile.write_text(json.dumps(inners))

    def to_json_directories(self, root: Path) -> None:
        root.mkdir(exist_ok=True, parents=True)
        for name, hparam in self.hparams.items():
            if hparam.kind is HparamKind.Categorical:
                outdir = root / "categorical"
            elif hparam.kind is HparamKind.Ordinal:
                outdir = root / "ordinal"
            elif hparam.kind is HparamKind.Continuous:
                outdir = root / "continuous"
            elif hparam.kind is HparamKind.Fixed:
                outdir = root / "fixed"
            else:
                raise ValueError(f"Impossible! Got kind: {hparam.kind}")
            outdir.mkdir(exist_ok=True, parents=True)
            path = outdir / f"{name}.json"
            hparam.to_json(path=path)

    @classmethod
    def from_json(cls: Type[Hparams], root: Path) -> Hparams:
        outfile = list(root.rglob("all_hparams.json"))[0]
        with open(outfile, "r") as handle:
            hpdicts: List[Dict[str, Any]] = json.load(handle)
        hparams = []
        for hpdict in hpdicts:
            ...
            if hpdict["kind"] == "categorical":
                hp = CategoricalHparam.from_dict(hpdict)
            elif hpdict["kind"] == "ordinal":
                hp = OrdinalHparam.from_dict(hpdict)
            elif hpdict["kind"] == "continuous":
                hp = ContinuousHparam.from_dict(hpdict)
            elif hpdict["kind"] == "fixed":
                hp = FixedHparam.from_dict(hpdict)
            else:
                raise ValueError("Impossible!")
            hparams.append(hp)

        return cls(hparams=hparams)

    @classmethod
    def from_json_directory(cls: Type[Hparams], root: Path) -> Hparams:
        jsons = sorted(root.rglob("*.json"))
        hparams = []
        for path in jsons:
            if path.parent.name == "categorical":
                hp = CategoricalHparam.from_json(path=path)
            elif path.parent.name == "ordinal":
                hp = OrdinalHparam.from_json(path=path)
            elif path.parent.name == "continuous":
                hp = ContinuousHparam.from_json(path=path)
            elif path.parent.name == "fixed":
                hp = FixedHparam.from_json(path=path)
            else:
                raise ValueError("Impossible!")
            hparams.append(hp)
        return cls(hparams=hparams)

    @classmethod
    def from_dicts(cls: Type[Hparams], tardicts: List[Dict[str, Any]]) -> Hparams:
        hparams = []
        for tardict in tardicts:
            if tardict["kind"] == "categorical":
                hp = CategoricalHparam.from_dict(hpdict=tardict)
            elif tardict["kind"] == "ordinal":
                hp = OrdinalHparam.from_dict(hpdict=tardict)
            elif tardict["kind"] == "continuous":
                hp = ContinuousHparam.from_dict(hpdict=tardict)
            elif tardict["kind"] == "fixed":
                hp = FixedHparam.from_dict(hpdict=tardict)
            else:
                raise ValueError("Impossible!")
            hparams.append(hp)
        return cls(hparams=hparams)

    def __eq__(self, __o: object) -> bool:
        o = __o
        if not isinstance(o, Hparams):
            raise ValueError("Can only find difference between instance of Hparams.")

        if sorted(self.hparams.keys()) != sorted(o.hparams.keys()):
            return False

        for name in sorted(self.hparams.keys()):
            hp1 = self.hparams[name]
            hp2 = o.hparams[name]
            if hp1 != hp2:
                return False
        return True

    def __sub__(self, o: Hparams) -> float:
        if not isinstance(o, Hparams):
            raise ValueError("Can only find difference between instance of Hparams.")
        if sorted(self.hparams.keys()) != sorted(o.hparams.keys()):
            names1 = set(self.hparams.keys())
            names2 = set(o.hparams.keys())
            diff = names1.symmetric_difference(names2)
            raise ValueError(
                "Cannot find difference between different sets of hparams. "
                f"Got left: {self.hparams} and right: {o.hparams}.\n\nDiffering "
                f"keys: {diff}"
            )
        diffs = []
        for name in sorted(self.hparams.keys()):
            hp1 = self.hparams[name]
            if hp1.kind is HparamKind.Fixed:
                continue
            hp2 = o.hparams[name]
            diffs.append(hp1 - hp2)
        return float(np.mean(diffs))

    def __str__(self) -> str:
        fmt = [f"{self.__class__.__name__}("]
        for name, hp in self.hparams.items():
            fmt.append(f"  {str(hp)}")
        fmt.append(")")
        return "\n".join(fmt)
