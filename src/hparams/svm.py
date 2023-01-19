from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from argparse import Namespace
from pathlib import Path
from typing import Any, Collection, Literal, Sequence

from numpy.random import Generator

from src.constants import SKLEARN_SGD_LR_DEFAULT as LR_DEFAULT
from src.constants import SKLEARN_SGD_LR_MAX as LR_MAX
from src.constants import SKLEARN_SGD_LR_MIN as LR_MIN
from src.constants import SVM_GAMMA as GAMMA
from src.enumerables import DatasetName
from src.hparams.hparams import (
    CategoricalHparam,
    ContinuousHparam,
    FixedHparam,
    Hparam,
    Hparams,
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

LINEAR_TUNED: dict[DatasetName, dict[str, Any] | None] = {
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

SGD_TUNED: dict[DatasetName, dict[str, Any] | None] = {
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


def svm_hparams(
    C: float | None = 1.0,
    gamma: float | None = GAMMA,
) -> list[Hparam]:
    # see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0088-0#Sec6
    # for a possible tuning range on C, gamma
    return [
        ContinuousHparam("C", C, max=1e5, min=1e-2, log_scale=True, default=1.0),
        ContinuousHparam(
            "gamma", gamma, max=1e3, min=1e-10, log_scale=True, default=GAMMA
        ),
        FixedHparam("kernel", value="rbf", default="rbf"),
    ]


def linear_svm_hparams(
    C: float | None = 1.0,
    penalty: Literal["l1", "l2"] = "l2",
    loss: Literal["hinge", "squared_hinge"] = "squared_hinge",
    dual: bool = True,
) -> list[Hparam]:
    # see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0088-0#Sec6
    # for a possible tuning range on C, gamma
    return [
        ContinuousHparam("C", C, max=1e5, min=1e-2, log_scale=True, default=1.0),
        CategoricalHparam(
            "loss", loss, categories=["hinge", "squared_hinge"], default="squared_hinge"
        ),
        CategoricalHparam("penalty", penalty, categories=["l1", "l2"], default="l2"),
        CategoricalHparam("dual", dual, categories=[True, False], default=True),
    ]


def sgd_svm_hparams(
    alpha: float | None = 1e-4,
    l1_ratio: float | None = 0.15,
    lr_init: float | None = 1e-3,
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
    loss: Literal["hinge", "modified_huber"] = "hinge",
    average: bool = False,
) -> list[Hparam]:
    # see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0088-0#Sec6
    # for a possible tuning range on C, gamma
    return [
        ContinuousHparam(
            "alpha", alpha, max=1e-1, min=1e-7, log_scale=True, default=1e-4
        ),
        ContinuousHparam(
            "l1_ratio", l1_ratio, max=1.0, min=0.0, log_scale=False, default=0.15
        ),
        ContinuousHparam(
            "eta0", lr_init, max=LR_MAX, min=LR_MIN, log_scale=True, default=LR_DEFAULT
        ),
        CategoricalHparam(
            "loss", loss, categories=["hinge", "modified_huber"], default="hinge"
        ),
        CategoricalHparam(
            "penalty",
            value=penalty,
            categories=["l1", "l2", "elasticnet", None],
            default="l2",
        ),
        CategoricalHparam("average", average, categories=[True, False], default=False),
        FixedHparam("learning_rate", value="adaptive"),
    ]


class SVMHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:

        if hparams is None:
            hparams = svm_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dsname: DatasetName) -> dict[str, Any]:
        hps = TUNED[dsname]
        if hps is None:
            return self.defaults().to_dict()
        return hps


class LinearSVMHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:

        if hparams is None:
            hparams = linear_svm_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dsname: DatasetName) -> dict[str, Any]:
        hps = LINEAR_TUNED[dsname]
        if hps is None:
            return self.defaults().to_dict()
        return hps

    def random(self, rng: Generator | None = None) -> LinearSVMHparams:
        hps = super().random(rng)
        assert isinstance(hps, LinearSVMHparams)
        if hps.is_valid():
            return hps
        return self.random(rng)

    def is_valid(self) -> bool:
        """
        Notes
        -----

        We get undocumented errors below:

            The combination of penalty='l1' and loss='squared_hinge' are not
            supported when dual=True

            The combination of penalty='l2' and loss='hinge' are not supported
            when dual=False

            The combination of penalty='l1' and loss='hinge' is not supported

        and so reject such random hparams.
        """
        hp = Namespace(**(self.to_dict()))
        if (hp.penalty == "l1") and (hp.loss == "squared_hinge") and (hp.dual is True):
            return False
        if (hp.penalty == "l2") and (hp.loss == "hinge") and (hp.dual is False):
            return False
        if (hp.penalty == "l1") and (hp.loss == "hinge"):
            return False
        return True


class SGDLinearSVMHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:

        if hparams is None:
            hparams = sgd_svm_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dsname: DatasetName) -> dict[str, Any]:
        hps = SGD_TUNED[dsname]
        if hps is None:
            return self.defaults().to_dict()
        return hps

    def random(self, rng: Generator | None = None) -> SGDLinearSVMHparams:
        hps = super().random(rng)
        assert isinstance(hps, SGDLinearSVMHparams)
        if hps.is_valid():
            return hps
        return self.random(rng)

    def is_valid(self) -> bool:
        """
        Notes
        -----

        We get undocumented errors below:

            The combination of penalty='l1' and loss='squared_hinge' are not
            supported when dual=True

            The combination of penalty='l2' and loss='hinge' are not supported
            when dual=False

            The combination of penalty='l1' and loss='hinge' is not supported

        and so reject such random hparams.
        """
        # hp = Namespace(**(self.to_dict()))
        # if (hp.penalty == "l1") and (hp.loss == "squared_hinge") and (hp.dual is True):
        #     return False
        # if (hp.penalty == "l2") and (hp.loss == "hinge") and (hp.dual is False):
        #     return False
        # if (hp.penalty == "l1") and (hp.loss == "hinge"):
        #     return False
        # return True
        return True
