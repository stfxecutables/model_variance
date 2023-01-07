from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Type

from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.hparams import Hparams
from src.models.dl_model import DLModel
from src.models.torch_base import MLP


class MLPModel(DLModel):
    def __init__(self, hparams: Hparams, runtime: RuntimeClass) -> None:
        super().__init__(hparams, runtime)
        self.kind: ClassifierKind = ClassifierKind.MLP
        self.model_cls: Type[MLP] = MLP
        self.model: MLP
