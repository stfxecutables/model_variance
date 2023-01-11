from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pathlib import Path
from typing import Any, Collection, Sequence

from src.enumerables import DatasetName
from src.hparams.hparams import (
    ContinuousHparam,
    FixedHparam,
    Hparam,
    Hparams,
    OrdinalHparam,
)

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


def xgboost_hparams(
    eta: float | None = None,
    lamda: float | None = None,
    alpha: float | None = None,
    num_round: int | None = None,
    gamma: float | None = None,
    colsample_bylevel: float | None = None,
    colsample_bynode: float | None = None,
    colsample_bytree: float | None = None,
    max_depth: int | None = None,
    max_delta_step: int | None = None,
    min_child_weight: float | None = None,
    subsample: float | None = None,
) -> list[Hparams]:
    """Note defaults are all None for XGBClassifier"""
    return [
        ContinuousHparam("eta", eta, max=1.0, min=0.001, log_scale=True),
        ContinuousHparam("lambda", lamda, max=1.0, min=1e-10, log_scale=True),
        ContinuousHparam("alpha", alpha, max=1.0, min=1e-10, log_scale=True),
        # XGB complains below are unused
        # OrdinalHparam("num_round", num_round, max=1000, min=1),
        ContinuousHparam("gamma", gamma, max=1.0, min=0.1, log_scale=True),
        ContinuousHparam(
            "colsample_bylevel", colsample_bylevel, max=1.0, min=0.1, log_scale=False
        ),
        ContinuousHparam(
            "colsample_bynode", colsample_bynode, max=1.0, min=0.1, log_scale=False
        ),
        ContinuousHparam(
            "colsample_bytree", colsample_bytree, max=1.0, min=0.1, log_scale=False
        ),
        OrdinalHparam("max_depth", max_depth, max=20, min=1),
        OrdinalHparam("max_delta_step", max_delta_step, max=10, min=0),
        ContinuousHparam(
            "min_child_weight", min_child_weight, max=20, min=0.1, log_scale=True
        ),
        ContinuousHparam("subsample", subsample, max=1.0, min=0.01, log_scale=False),
        FixedHparam("enable_categorical", value=True),
        FixedHparam("tree_method", value="hist"),
    ]


class XGBoostHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:

        if hparams is None:
            hparams = xgboost_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dsname: DatasetName) -> dict[str, Any]:
        hps = TUNED[dsname]
        if hps is None:
            return self.defaults().to_dict()
        return hps


if __name__ == "__main__":
    xgbs = XGBoostHparams()
    print(xgbs)
    print(xgbs)
