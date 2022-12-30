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


def test_evaluator() -> None:
    try:
        tempdir = Path(mkdtemp(dir=DIR))
        out = DIR / f"{tempdir.name}/evaluator"
        out.mkdir(exist_ok=True, parents=True)
        for _ in range(50):
            hp_cls = choice([SVMHparams, XGBoostHparams])
            ds = choice([*DatasetName])
            classifier = (
                ClassifierKind.XGBoost if hp_cls is XGBoostHparams else ClassifierKind.SVM
            )
            hps = hp_cls().random()
            dim_reduce = choice([25, 50, 75, None])
            cont_perturb = choice([*DataPerturbation, None])
            cat_perturb = choice([None, 0.1, 0.2])
            h_perturb = choice([25, 50, 75, None])
            train_downsample = choice([25, 50, 75, None])
            cat_perturb_level = choice(["sample", "label"])
            ev = Evaluator(
                dataset_name=ds,
                classifier_kind=classifier,
                hparams=hps,
                dimension_reduction=dim_reduce,
                continuous_perturb=cont_perturb,
                categorical_perturb=cat_perturb,
                hparam_perturb=h_perturb,
                train_downsample=train_downsample,
                categorical_perturb_level=cat_perturb_level,
            )
            ev.to_json(out)
            ev2 = Evaluator.from_json(out)
            if ev != ev2:
                info = missing_keys(ev, ev2)
                if info is not None:
                    raise ValueError(
                        f"Objects differ:\n {info}\n" f"left: {ev}\n" f"right: {ev2}\n"
                    )
            rmtree(out)
    except Exception as e:
        raise e
    finally:
        rmtree(tempdir)
