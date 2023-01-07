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

import numpy as np

from src.enumerables import ClassifierKind, DataPerturbation, DatasetName
from src.evaluator import Evaluator
from src.hparams.hparams import (
    CategoricalHparam,
    ContinuousHparam,
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
        for _ in range(1000):
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
                repeat=int(np.random.randint(0, 50)),
                run=int(np.random.randint(0, 50)),
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
