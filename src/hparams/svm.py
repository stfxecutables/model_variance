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


def svm_hparams(
    C: float | None = None,
    gamma: float | None = None,
) -> list[Hparams]:
    # see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0088-0#Sec6
    # for a possible tuning range on C, gamma
    return [
        ContinuousHparam("C", C, max=1e5, min=1e-2, log_scale=True),
        ContinuousHparam("gamma", gamma, max=1e3, min=1e-10, log_scale=True),
    ]


class SVMHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:

        if hparams is None:
            hparams = svm_hparams()
        super().__init__(hparams)
