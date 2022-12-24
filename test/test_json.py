# from __future__ import annotations

# # fmt: off
# import sys  # isort:skip
# from pathlib import Path  # isort: skip
# ROOT = Path(__file__).resolve().parent.parent  # isort: skip
# sys.path.append(str(ROOT))  # isort: skip
# # fmt: on


from pathlib import Path
from random import choice
from shutil import rmtree
from tempfile import mkdtemp
from uuid import uuid4

import numpy as np

from src.hparams.hparams import (
    CategoricalHparam,
    ContinuousHparam,
    Hparam,
    Hparams,
    OrdinalHparam,
)
from src.hparams.svm import SVMHparams
from src.hparams.xgboost import XGBoostHparams

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


def random_hparams() -> list[Hparam]:
    n = choice(list(range(1, 100)))
    choices = [random_categorical, random_continuous, random_ordinal]
    return [choice(choices)() for _ in range(n)]


def test_hparam_cont() -> None:
    tempdir = Path(mkdtemp(dir=DIR))
    out = DIR / f"{tempdir.name}/test.json"
    try:
        for _ in range(50):
            c = random_continuous()
            c.to_json(out)
            c2 = ContinuousHparam.from_json(out)
            assert c.__dict__ == c2.__dict__
    except Exception as e:
        raise e
    finally:
        rmtree(tempdir)


def test_hparam_cat() -> None:
    tempdir = Path(mkdtemp(dir=DIR))
    # generate a-zA-Z
    out = DIR / f"{tempdir.name}/test.json"
    try:
        for _ in range(50):
            c = random_categorical()
            c.to_json(out)
            c2 = CategoricalHparam.from_json(out)
            assert c.__dict__ == c2.__dict__
    except Exception as e:
        raise e
    finally:
        rmtree(tempdir)


def test_hparam_ordinal() -> None:
    tempdir = Path(mkdtemp(dir=DIR))
    out = DIR / f"{tempdir.name}/test.json"
    try:
        for _ in range(50):
            c = random_ordinal()
            c.to_json(out)
            c2 = OrdinalHparam.from_json(out)
            assert c.__dict__ == c2.__dict__
    except Exception as e:
        raise e
    finally:
        rmtree(tempdir)


def test_hparams_list() -> None:
    try:
        tempdir = Path(mkdtemp(dir=DIR))
        outdir = DIR / f"{tempdir.name}/hps"
        for _ in range(50):
            hps = Hparams(random_hparams())
            hps.to_json(root=outdir)
            hp = Hparams.from_json(root=outdir)
            assert hps - hp < 1e-14
            assert hp - hps < 1e-14
            rmtree(outdir)
    except Exception as e:
        raise e
    finally:
        rmtree(tempdir)


def test_hparams_xgb() -> None:
    try:
        tempdir = Path(mkdtemp(dir=DIR))
        outdir = DIR / f"{tempdir.name}/hps"
        for _ in range(50):
            hps = XGBoostHparams().random()
            hps.to_json(root=outdir)
            hp = XGBoostHparams.from_json(root=outdir)
            assert hps - hp < 1e-14
            assert hp - hps < 1e-14
            rmtree(outdir)
    except Exception as e:
        raise e
    finally:
        rmtree(tempdir)

def test_hparams_svm() -> None:
    try:
        tempdir = Path(mkdtemp(dir=DIR))
        outdir = DIR / f"{tempdir.name}/hps"
        for _ in range(50):
            hps = SVMHparams().random()
            hps.to_json(root=outdir)
            hp = SVMHparams.from_json(root=outdir)
            assert hps - hp < 1e-14
            assert hp - hps < 1e-14
            rmtree(outdir)
    except Exception as e:
        raise e
    finally:
        rmtree(tempdir)
