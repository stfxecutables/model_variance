from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Sequence, Union

from src.constants import DROPOUT_DEFAULT as P
from src.constants import MLP_LR_INIT_DEFAULT as LR
from src.constants import MLP_WD_DEFAULT as WD
from src.constants import MLP_WIDTH_DEFAULT as WIDTH
from src.enumerables import DatasetName
from src.hparams.hparams import ContinuousHparam, Hparam, Hparams, OrdinalHparam

TUNED: Dict[DatasetName, Optional[Dict[str, Any]]] = {
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


def mlp_hparams(
    lr: Optional[float] = None,
    wd: Optional[float] = None,
    dropout: Optional[float] = None,
    w1: Optional[int] = None,
    w2: Optional[int] = None,
) -> List[Hparam]:
    return [
        ContinuousHparam("lr", lr, max=5e-1, min=1e-5, default=LR, log_scale=True),
        ContinuousHparam("wd", wd, max=5e-1, min=1e-8, default=WD, log_scale=True),
        ContinuousHparam(
            "dropout", dropout, min=0.0, max=0.8, default=P, log_scale=False
        ),
        OrdinalHparam("width1", w1, max=1024, min=32, default=WIDTH),
        OrdinalHparam("width2", w2, max=1024, min=32, default=WIDTH),
    ]


class MLPHparams(Hparams):
    def __init__(
        self,
        hparams: Optional[Union[Collection[Hparam], Sequence[Hparam]]] = None,
    ) -> None:

        if hparams is None:
            hparams = mlp_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dsname: DatasetName) -> Dict[str, Any]:
        hps = TUNED[dsname]
        if hps is None:
            return self.defaults().to_dict()
        return hps
