from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from argparse import Namespace
from pathlib import Path
from typing import Any, Collection, Dict, List, Literal, Optional, Sequence, Union

from numpy.random import Generator
from typing_extensions import Literal

from src.constants import SKLEARN_SGD_LR_DEFAULT as LR_DEFAULT
from src.constants import SKLEARN_SGD_LR_MAX as LR_MAX
from src.constants import SKLEARN_SGD_LR_MIN as LR_MIN
from src.constants import SVM_GAMMA as GAMMA
from src.enumerables import ClassifierKind, DatasetName
from src.hparams.hparams import (
    CategoricalHparam,
    ContinuousHparam,
    FixedHparam,
    Hparam,
    Hparams,
    OrdinalHparam,
)


def classic_svm_hparams(
    C: Optional[float] = 1.0,
    gamma: Optional[float] = GAMMA,
) -> List[Hparam]:
    # see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0088-0#Sec6
    # for a possible tuning range on C, gamma
    return [
        ContinuousHparam("C", C, max=1e5, min=1e-5, log_scale=True, default=1.0),
        ContinuousHparam(
            "gamma", gamma, max=1e3, min=1e-10, log_scale=True, default=GAMMA
        ),
        FixedHparam("kernel", value="rbf", default="rbf"),
    ]


def linear_svm_hparams(
    C: Optional[float] = 1.0,
    penalty: Literal["l1", "l2"] = "l2",
    loss: Literal["hinge", "squared_hinge"] = "squared_hinge",
    dual: bool = True,
) -> List[Hparam]:
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
    alpha: Optional[float] = 1e-4,
    l1_ratio: Optional[float] = 0.15,
    lr_init: Optional[float] = 1e-3,
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
    loss: Literal["hinge", "modified_huber"] = "hinge",
    average: bool = False,
) -> List[Hparam]:
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


def nystroem_hparams(
    gamma: Optional[float] = 0.1,
    n_components: Optional[int] = 20,
    alpha: Optional[float] = 1e-4,
    l1_ratio: Optional[float] = 0.15,
    lr_init: Optional[float] = 1e-3,
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
    average: bool = False,
) -> List[Hparam]:
    # see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0088-0#Sec6
    # for a possible tuning range on C, gamma
    return [
        ContinuousHparam("gamma", gamma, max=1e3, min=1e-10, log_scale=True, default=0.1),
        OrdinalHparam("n_components", n_components, max=50, min=1, default=20),
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
            "penalty",
            value=penalty,
            categories=["l1", "l2", "elasticnet", None],
            default="l2",
        ),
        CategoricalHparam("average", average, categories=[True, False], default=False),
        FixedHparam("loss", value="hinge", default="hinge"),
        FixedHparam("learning_rate", value="adaptive", default="adaptive"),
        FixedHparam("n_jobs", value=1, default=1),
    ]


class ClassicSVMHparams(Hparams):
    kind = ClassifierKind.SVM

    def __init__(
        self,
        hparams: Optional[Union[Collection[Hparam], Sequence[Hparam]]] = None,
    ) -> None:

        if hparams is None:
            hparams = classic_svm_hparams()
        super().__init__(hparams)



class LinearSVMHparams(Hparams):
    kind = ClassifierKind.LinearSVM

    def __init__(
        self,
        hparams: Optional[Union[Collection[Hparam], Sequence[Hparam]]] = None,
    ) -> None:

        if hparams is None:
            hparams = linear_svm_hparams()
        super().__init__(hparams)


    def random(self, rng: Optional[Generator] = None) -> LinearSVMHparams:
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
    kind = ClassifierKind.SGD_SVM

    def __init__(
        self,
        hparams: Optional[Union[Collection[Hparam], Sequence[Hparam]]] = None,
    ) -> None:

        if hparams is None:
            hparams = sgd_svm_hparams()
        super().__init__(hparams)


    def random(self, rng: Optional[Generator] = None) -> SGDLinearSVMHparams:
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


class NystroemHparams(Hparams):
    kind = ClassifierKind.NystroemSVM

    def __init__(
        self,
        hparams: Union[Collection[Hparam], Sequence[Hparam], None] = None,
    ) -> None:
        if hparams is None:
            hparams = nystroem_hparams()
        super().__init__(hparams)

    def set_n_jobs(self, n_jobs: int) -> None:
        self.hparams["n_jobs"].value = n_jobs

    def ny_dict(self) -> Dict[str, Any]:
        full = self.to_dict()
        d = {"gamma": full["gamma"], "n_components": full["n_components"]}
        return d

    def sgd_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop("gamma")
        d.pop("n_components")
        return d
