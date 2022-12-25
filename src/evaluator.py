from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
from enum import Enum, EnumMeta
from typing import Literal

from src.classifier import Classifier
from src.dataset import Dataset
from src.enumerables import DataPerturbation, DatasetName, HparamPerturbation
from src.hparams.hparams import Hparam, Hparams
from src.serialize import DirJSONable, FileJSONable

Percentage = Literal[25, 50, 75]

def value_or_none(enumerable: Enum | None) -> Enum | None:
    if isinstance(enumerable, EnumMeta):
        raise TypeError("Must pass Enum instance, not class.")
    if isinstance(enumerable, Enum):
        return enumerable.value
    return None

class Evaluator(DirJSONable):
    """For performing one repeat (i.e. `r` validation runs) with the sources
    of variance given in the constructor

    Parameters
    ----------
    dataset: Dataset
        Base data

    classifier: Classifier
        Model to fit

    train_downsample: float | None
        Proportion `p` in (0, 1] to down*sample* to, e.g. reduce the training
        sample to. E.g. len(X_train) = ceil(len(X) * p). Should be a value in
        [0.30, 0.45, 0.60, 0.75, 0.90].


    """

    def __init__(
        self,
        dataset_name: DatasetName,
        classifier: Classifier,
        hparams: Hparams,
        dimension_reduction: Percentage | None,
        continuous_perturb: DataPerturbation | None,
        categorical_perturb: float | None,
        hparam_perturb: HparamPerturbation | None,
        train_downsample: Percentage | None,
        categorical_perturb_level: Literal["sample", "label"] = "label",
    ) -> None:
        self.dataset_name: DatasetName = dataset_name
        self.dataset_: Dataset | None = None
        self.classifer: Classifier = classifier
        self.hparams: Hparams = hparams
        self.dimension_reduction: Percentage | None = dimension_reduction
        self.continuous_perturb: DataPerturbation | None = continuous_perturb
        self.categorical_perturb: float | None = categorical_perturb
        self.hparam_perturb: HparamPerturbation | None = hparam_perturb
        self.train_downsample: Percentage | None = train_downsample
        self.categorical_perturb_level: Literal[
            "sample", "label"
        ] = categorical_perturb_level

    @property
    def dataset(self) -> Dataset:
        if self.dataset_ is None:
            self.dataset_ = Dataset(self.dataset_name)
        return self.dataset_

    def to_json(self, root: Path) -> None:
        root.mkdir(exist_ok=True, parents=True)
        hps = root / "hparams"
        out = root / "evaluator.json"

        self.hparams.to_json(hps)
        with open(out, "w") as fp:
            json.dump({
                "dataset_name": self.dataset_name.value,
                "classifier": self.classifer,
                "dimension_reduction": self.dimension_reduction,
                "continuous_perturb": value_or_none(self.continuous_perturb),
                "categorical_perturb": self.categorical_perturb,
                "hparam_perturb": value_or_none(self.hparam_perturb),
                "train_downsample": self.train_downsample,
                "categorical_perturb_level": self.categorical_perturb_level,
            }, fp)

    @classmethod
    def from_json(cls: Evaluator, root: Path) -> Evaluator:
        root.mkdir(exist_ok=True, parents=True)
        hps = root / "hparams"
        out = root / "evaluator.json"
        hparams = ...
