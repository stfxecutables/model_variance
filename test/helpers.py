from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from numpy.random import Generator

from src.hparams.hparams import (
    CategoricalHparam,
    ContinuousHparam,
    FixedHparam,
    Hparam,
    OrdinalHparam,
)

ROOT = Path(__file__).resolve().parent.parent  # isort: skip
DIR = ROOT / "__test_temp__"
DIR.mkdir(exist_ok=True)
CATS = [chr(i) for i in (list(range(97, 123)) + list(range(65, 91)))]


def random_continuous(rng: Optional[Generator] = None) -> ContinuousHparam:
    if rng is None:
        rng = np.random.default_rng()
    log_scale = bool(rng.choice([True, False]))
    value = float(rng.uniform(0, 1))
    hsh = rng.bytes(8).hex()
    mn = 1e-15 if log_scale else 0.0
    return ContinuousHparam(
        f"test_float_{hsh}", value=value, max=1.0, min=mn, log_scale=log_scale
    )


def random_fixed(rng: Optional[Generator] = None) -> FixedHparam[float]:
    if rng is None:
        rng = np.random.default_rng()
    value = rng.uniform(0, 1)
    hsh = rng.bytes(8).hex()
    return FixedHparam(f"test_fixed_{hsh}", value=value)


def random_categorical(rng: Optional[Generator] = None) -> CategoricalHparam:
    if rng is None:
        rng = np.random.default_rng()
    size = rng.integers(1, 20)
    values = rng.choice(CATS, size=size, replace=False)
    value = rng.choice(values)
    hsh = rng.bytes(8).hex()
    return CategoricalHparam(f"test_cat_{hsh}", value=value, categories=values)


def random_ordinal(rng: Optional[Generator] = None) -> OrdinalHparam:
    if rng is None:
        rng = np.random.default_rng()
    mx = rng.integers(1, 500)
    value = rng.choice(list(range(mx)))
    hsh = rng.bytes(8).hex()
    return OrdinalHparam(f"test_ord_{hsh}", value=value, min=0, max=mx)


def random_hparams(rng: Optional[Generator] = None) -> List[Hparam[Any, Any]]:
    if rng is None:
        rng = np.random.default_rng()
    n = rng.choice(list(range(1, 100)))
    choices = [random_categorical, random_continuous, random_ordinal, random_fixed]
    return [rng.choice(choices)(rng) for _ in range(n)]
