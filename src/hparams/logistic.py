from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pathlib import Path
from typing import Any, Collection, Sequence

from src.constants import LR_LR_INIT_DEFAULT as LR
from src.constants import LR_WD_DEFAULT as WD
from src.enumerables import DatasetName
from src.hparams.hparams import ContinuousHparam, Hparam, Hparams

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


def lr_hparams(
    lr: float | None = None,
    wd: float | None = None,
) -> list[Hparams]:
    return [
        ContinuousHparam("lr", lr, max=5e-1, min=1e-5, default=LR, log_scale=True),
        ContinuousHparam("wd", wd, max=5e-1, min=1e-8, default=WD, log_scale=True),
    ]


class LRHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:

        if hparams is None:
            hparams = lr_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dsname: DatasetName) -> dict[str, Any]:
        hps = TUNED[dsname]
        if hps is None:
            return self.defaults().to_dict()
        return hps

