from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Literal

from src.classifier import Classifier
from src.dataset import Dataset
from src.enumerables import DataPerturbation, DatasetName, HparamPerturbation
from src.hparams.hparams import Hparam

Percentage = Literal[25, 50, 75]


class Evaluator:
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