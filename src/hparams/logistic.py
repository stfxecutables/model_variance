from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pathlib import Path
from typing import Collection, Sequence

from src.hparams.hparams import ContinuousHparam, Hparam, Hparams


def lr_hparams(
    lr: float | None = None,
    wd: float | None = None,
) -> list[Hparams]:
    return [
        ContinuousHparam("lr", lr, max=5e-1, min=1e-5, log_scale=True),
        ContinuousHparam("wd", wd, max=5e-1, min=1e-8, log_scale=True),
    ]


class LRHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:

        if hparams is None:
            hparams = lr_hparams()
        super().__init__(hparams)
