# from __future__ import annotations

# # fmt: off
# import sys  # isort:skip
# from pathlib import Path  # isort: skip
# ROOT = Path(__file__).resolve().parent.parent  # isort: skip
# sys.path.append(str(ROOT))  # isort: skip
# # fmt: on


from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from src.hparams.hparams import CategoricalHparam, ContinuousHparam, OrdinalHparam

ROOT = Path(__file__).resolve().parent.parent  # isort: skip
DIR = ROOT / "__test_temp__"
DIR.mkdir(exist_ok=True)


def test_hparam_cont() -> None:
    tempdir = Path(mkdtemp(dir=DIR))
    out = DIR / f"{tempdir.name}/test.json"
    try:
        for log_scale in [True, False]:
            for _ in range(50):
                value = np.random.uniform(0, 1)
                c = ContinuousHparam(
                    "test_float", value=value, max=1.0, min=0.0, log_scale=log_scale
                )
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
    cats = [chr(i) for i in (list(range(97, 123)) + list(range(65, 91)))]
    out = DIR / f"{tempdir.name}/test.json"
    try:
        for _ in range(50):
            size = np.random.randint(1, 20)
            values = np.random.choice(cats, size=size, replace=False)
            value = np.random.choice(values)
            c = CategoricalHparam("test_cat", value=value, categories=values)
            c.to_json(out)
            c2 = CategoricalHparam.from_json(out)
            assert c.__dict__ == c2.__dict__
    except Exception as e:
        raise e
    finally:
        rmtree(tempdir)

def test_hparam_ordinal() -> None:
    tempdir = Path(mkdtemp(dir=DIR))
    # generate a-zA-Z
    out = DIR / f"{tempdir.name}/test.json"
    try:
        for _ in range(50):
            mx = np.random.randint(1, 500)
            value = np.random.choice(list(range(mx)))
            c = OrdinalHparam("test_ord", value=value, min=0, max=mx)
            c.to_json(out)
            c2 = OrdinalHparam.from_json(out)
            assert c.__dict__ == c2.__dict__
    except Exception as e:
        raise e
    finally:
        rmtree(tempdir)
