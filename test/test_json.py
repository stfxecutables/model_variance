# from __future__ import annotations

# # fmt: off
# import sys  # isort:skip
# from pathlib import Path  # isort: skip
# ROOT = Path(__file__).resolve().parent.parent  # isort: skip
# sys.path.append(str(ROOT))  # isort: skip
# # fmt: on


from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

from src.hparams.hparams import CategoricalHparam, ContinuousHparam, OrdinalHparam

ROOT = Path(__file__).resolve().parent.parent  # isort: skip
DIR = ROOT / "__test_temp__"
DIR.mkdir(exist_ok=True)


def test_hparam_cont() -> None:
    tempdir = TemporaryDirectory(dir=DIR)
    out = DIR / f"{tempdir.name}/test.json"
    try:
        c = ContinuousHparam("test_float", value=0.2, max=1.0, min=0.0, log_scale=False)
        c.to_json(out)
        c2 = ContinuousHparam.from_json(out)
        assert c.__dict__ == c2.__dict__
    except Exception as e:
        raise e
    finally:
        tempdir.close()
