from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Any, Mapping, Type

from numpy import ndarray

from src.constants import LR_MAX_EPOCHS
from src.dataset import Dataset
from src.enumerables import ClassifierKind, DatasetName, RuntimeClass
from src.hparams.logistic import LRHparams
from src.models.dl_model import DLModel
from src.models.torch_base import LogisticRegression

TUNED: dict[DatasetName, dict[str, Any] | None] = {
    DatasetName.Arrhythmia: None,
    DatasetName.Kc1: None,
    DatasetName.ClickPrediction: None,
    DatasetName.BankMarketing: None,
    DatasetName.BloodTransfusion: None,
    DatasetName.Cnae9: None,
    DatasetName.Ldpa: None,
    DatasetName.Nomao: None,
    DatasetName.Phoneme: None,
    DatasetName.SkinSegmentation: None,
    DatasetName.WalkingActivity: None,
    DatasetName.Adult: None,
    DatasetName.Higgs: None,
    DatasetName.Numerai28_6: None,
    DatasetName.Kr_vs_kp: None,
    DatasetName.Connect4: None,
    DatasetName.Shuttle: None,
    # DevnagariScript = "devnagari-script: None"
    DatasetName.Car: None,
    DatasetName.Australian: None,
    DatasetName.Segment: None,
    # FashionMnist = "fashion-mnist: None"
    DatasetName.JungleChess: None,
    DatasetName.Christine: None,
    DatasetName.Jasmine: None,
    DatasetName.Sylvine: None,
    DatasetName.Miniboone: None,
    DatasetName.Dilbert: None,
    DatasetName.Fabert: None,
    DatasetName.Volkert: None,
    DatasetName.Dionis: None,
    DatasetName.Jannis: None,
    DatasetName.Helena: None,
    DatasetName.Aloi: None,
    DatasetName.CreditCardFraud: None,
    DatasetName.Credit_g: None,
    DatasetName.Anneal: None,
    DatasetName.MfeatFactors: None,
    DatasetName.Vehicle: None,
}


class LRModel(DLModel):
    def __init__(self, hparams: LRHparams, dataset: Dataset, logdir: Path) -> None:
        super().__init__(hparams=hparams, dataset=dataset, logdir=logdir)
        self.max_epochs = LR_MAX_EPOCHS
        self.kind: ClassifierKind = ClassifierKind.LR
        self.hparams: LRHparams
        self.model_cls: Type[LogisticRegression] = LogisticRegression
        self.model: LogisticRegression
