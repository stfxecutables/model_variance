from pathlib import Path
from random import choice
from typing import Any
from uuid import uuid4

import numpy as np

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


def random_continuous() -> ContinuousHparam:
    log_scale = choice([True, False])
    value = np.random.uniform(0, 1)
    hsh = uuid4().hex
    mn = 1e-15 if log_scale else 0.0
    return ContinuousHparam(
        f"test_float_{hsh}", value=value, max=1.0, min=mn, log_scale=log_scale
    )


def random_fixed() -> FixedHparam[float]:
    value = np.random.uniform(0, 1)
    hsh = uuid4().hex
    return FixedHparam(f"test_fixed_{hsh}", value=value)


def random_categorical() -> CategoricalHparam:
    size = np.random.randint(1, 20)
    values = np.random.choice(CATS, size=size, replace=False)
    value = np.random.choice(values)
    hsh = uuid4().hex
    return CategoricalHparam(f"test_cat_{hsh}", value=value, categories=values)


def random_ordinal() -> OrdinalHparam:
    mx = np.random.randint(1, 500)
    value = np.random.choice(list(range(mx)))
    hsh = uuid4().hex
    return OrdinalHparam(f"test_ord_{hsh}", value=value, min=0, max=mx)


def random_hparams() -> list[Hparam[Any, Any]]:
    n = choice(list(range(1, 100)))
    choices = [random_categorical, random_continuous, random_ordinal, random_fixed]
    return [choice(choices)() for _ in range(n)]
