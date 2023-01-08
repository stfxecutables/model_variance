from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Mapping, Type

from src.constants import MLP_MAX_EPOCHS
from src.dataset import Dataset
from src.enumerables import ClassifierKind, RuntimeClass
from src.hparams.mlp import MLPHparams
from src.models.dl_model import DLModel
from src.models.torch_base import MLP


class MLPModel(DLModel):
    def __init__(self, hparams: MLPHparams, dataset: Dataset, logdir: Path) -> None:
        self.max_epochs = MLP_MAX_EPOCHS
        super().__init__(hparams=hparams, dataset=dataset, logdir=logdir)
        self.kind: ClassifierKind = ClassifierKind.MLP
        self.hparams: MLPHparams
        self.model_cls: Type[MLP] = MLP
        self.model: MLP
