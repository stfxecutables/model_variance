from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
import os
import sys
import traceback
from argparse import Namespace
from base64 import urlsafe_b64encode
from enum import Enum, EnumMeta
from shutil import rmtree
from time import strftime
from typing import Any, Literal, Type, TypeVar, overload
from uuid import uuid4

import numpy as np
from numpy import ndarray

from src.constants import DEBUG_LOGS, LOGS, ensure_dir
from src.dataset import Dataset
from src.enumerables import (
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
)
from src.hparams.hparams import Hparams
from src.models.logistic import LRModel, SGDLRModel
from src.models.mlp import MLPModel
from src.models.model import ClassifierModel
from src.models.svc import LinearSVCModel, SGDLinearSVCModel, SVCModel
from src.models.xgb import XGBoostModel
from src.serialize import DirJSONable
from src.utils import missing_keys

Percentage = Literal[25, 50, 75]
E = TypeVar("E")


def value_or_none(enumerable: Enum | None) -> Enum | None:
    if isinstance(enumerable, EnumMeta):
        raise TypeError("Must pass Enum instance, not class.")
    if isinstance(enumerable, Enum):
        return enumerable.value
    return None


def to_enum_or_none(enum_type: Type[E] | None, value: str | None) -> E | None:
    if not isinstance(enum_type, EnumMeta):
        raise TypeError("Must pass enum class, not instance.")
    if value is None:
        return None
    return enum_type(value)


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
        classifier_kind: ClassifierKind,
        repeat: int,
        run: int,
        hparams: Hparams,
        dimension_reduction: Percentage | None,
        continuous_perturb: DataPerturbation | None,
        categorical_perturb: float | None,
        hparam_perturb: HparamPerturbation | None,
        train_downsample: Percentage | None,
        categorical_perturb_level: Literal["sample", "label"] = "label",
        debug: bool = False,
        _suppress_json: bool = False,
    ) -> None:
        self.dataset_name: DatasetName = dataset_name
        self.dataset_: Dataset | None = None
        self.classifer_kind: ClassifierKind = classifier_kind
        self.repeat: int = repeat
        self.run: int = run
        self.hparams: Hparams = hparams
        self.dimension_reduction: Percentage | None = dimension_reduction
        self.continuous_perturb: DataPerturbation | None = continuous_perturb
        self.categorical_perturb: float | None = categorical_perturb
        self.hparam_perturb: HparamPerturbation | None = hparam_perturb
        self.train_downsample: Percentage | None = train_downsample
        self._model: ClassifierModel | None = None
        self.categorical_perturb_level: Literal[
            "sample", "label"
        ] = categorical_perturb_level
        self.debug = debug
        self.logdir = self.get_logdir()
        if not _suppress_json:
            self.to_json(self.logdir)
        if not debug:
            print(f"Evaluator results will be logged to {self.logdir}")

    @property
    def model(self) -> ClassifierModel:
        if self._model is not None:
            return self._model
        kind = self.classifer_kind
        logdir = (
            self.logdir
            if kind
            in [
                ClassifierKind.LR,
                ClassifierKind.SGD_LR,
                ClassifierKind.SVM,
                ClassifierKind.SGD_SVM,
                ClassifierKind.LinearSVM,
                ClassifierKind.XGBoost,
            ]
            else self.dl_dir
        )
        args: dict[str, Any] = dict(
            hparams=self.hparams,
            dataset=self.dataset,
            logdir=logdir,
        )
        if kind is ClassifierKind.LR:
            self._model = LRModel(**args)
        elif kind is ClassifierKind.SGD_LR:
            self._model = SGDLRModel(**args)
        elif kind is ClassifierKind.SVM:
            self._model = SVCModel(**args)
        elif kind is ClassifierKind.LinearSVM:
            self._model = LinearSVCModel(**args)
        elif kind is ClassifierKind.SGD_SVM:
            self._model = SGDLinearSVCModel(**args)
        elif kind is ClassifierKind.XGBoost:
            self._model = XGBoostModel(**args)
        elif kind is ClassifierKind.MLP:
            self._model = MLPModel(**args)
        else:
            raise ValueError(f"Unknown model kind: {self.classifer_kind}")
        return self._model

    def get_logdir(self) -> Path:
        c = self.classifer_kind.value
        d = self.dataset_name.value
        dim = self.dimension_reduction
        red = "full" if dim is None else f"reduce={dim}"

        rep = f"rep={self.repeat:03d}"
        run = f"run={self.run:03d}"
        jid = os.environ.get("SLURM_JOB_ID")
        aid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if jid is None:
            slurm_id = None
        elif aid is not None:
            slurm_id = f"{jid}_{aid}"
        else:
            slurm_id = jid

        ts = strftime("%b-%d--%H-%M-%S")
        hsh = urlsafe_b64encode(uuid4().bytes).decode()
        uid = f"{ts}__{hsh}" if slurm_id is None else f"{slurm_id}__{ts}__{hsh}"
        root = DEBUG_LOGS if self.debug else LOGS
        return ensure_dir(root / f"{c}/{d}/{red}/{rep}/{run}/{uid}")

    @property
    def res_dir(self) -> Path:
        """This a dynamic property to ensure it matches `self.logdir` on deserialization"""
        return ensure_dir(self.logdir / "results")

    @property
    def preds_dir(self) -> Path:
        return ensure_dir(self.logdir / "preds")

    @property
    def metrics_dir(self) -> Path:
        return ensure_dir(self.logdir / "metrics")

    @property
    def dl_dir(self) -> Path:
        return ensure_dir(self.logdir / "dl_logs")

    @property
    def ckpt_file(self) -> Path:
        """This should be a unique"""
        run = self.logdir.parent
        fname = self.logdir.name
        # do NOT ensure it exists: we create it after saving preds
        return run / f"{fname}.ckpt"

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
            json.dump(
                {
                    "dataset_name": self.dataset_name.value,
                    "classifier_kind": self.classifer_kind.value,
                    "dimension_reduction": self.dimension_reduction,
                    "continuous_perturb": value_or_none(self.continuous_perturb),
                    "categorical_perturb": self.categorical_perturb,
                    "hparam_perturb": value_or_none(self.hparam_perturb),
                    "train_downsample": self.train_downsample,
                    "categorical_perturb_level": self.categorical_perturb_level,
                    "repeat": self.repeat,
                    "run": self.run,
                    "debug": self.debug,
                },
                fp,
                indent=2,
            )

    @classmethod
    def from_json(cls: Type[Evaluator], root: Path) -> Evaluator:
        root.mkdir(exist_ok=True, parents=True)
        hps = root / "hparams"
        out = root / "evaluator.json"
        hparams = Hparams.from_json(hps)
        with open(out, "r") as fp:
            d = Namespace(**json.load(fp))

        c_perturb = to_enum_or_none(DataPerturbation, d.continuous_perturb)
        h_perturb = to_enum_or_none(HparamPerturbation, d.hparam_perturb)

        new = cls(
            dataset_name=DatasetName(d.dataset_name),
            classifier_kind=ClassifierKind(d.classifier_kind),
            hparams=hparams,
            dimension_reduction=d.dimension_reduction,
            continuous_perturb=c_perturb,
            categorical_perturb=d.categorical_perturb,
            hparam_perturb=h_perturb,
            train_downsample=d.train_downsample,
            categorical_perturb_level=d.categorical_perturb_level,
            repeat=d.repeat,
            run=d.run,
            debug=d.debug,
            _suppress_json=True,
        )
        new.logdir = root
        return new

    @overload
    def evaluate(
        self, no_pred: bool = False, return_test_acc: Literal[True] = True
    ) -> float:
        ...

    @overload
    def evaluate(
        self, no_pred: bool = False, return_test_acc: Literal[False] = False
    ) -> None:
        ...

    def evaluate(
        self, no_pred: bool = False, return_test_acc: bool = False
    ) -> float | None:
        if self.preds_dir.exists():
            with os.scandir(self.preds_dir) as files:
                if next(files, None) is not None:
                    raise FileExistsError(
                        "Impossible! Prediction data with same JOBID or hash "
                        f"already present in {self.res_dir}."
                    )
        ckpt_dir = self.ckpt_file.parent
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if len(ckpts) > 0:
                raise FileExistsError(
                    f"Checkpoint file {self.ckpt_file} exists, and suggests pred "
                    f"data for this evaluation repeat and run:\n{self}\n"
                    f"is already present in {self.logdir}."
                )
        try:
            ds = self.dataset
            # if self.classifer_kind in [ClassifierKind.MLP, ClassifierKind.LR]:
            #     raise NotImplementedError()
            X_train, y_train, X_test, y_test = ds.get_monte_carlo_splits(
                train_downsample=self.train_downsample,
                cont_perturb=self.continuous_perturb,
                cat_perturb_prob=self.categorical_perturb,
                cat_perturb_level=self.categorical_perturb_level,
                reduction=self.dimension_reduction,
                repeat=self.repeat,
                run=self.run,
            )
            self.model.fit(X=X_train, y=y_train)
            if return_test_acc:
                preds, targs = self.model.predict(X=X_test, y=y_test)
                if preds.ndim == 2:
                    return float(np.mean(np.argmax(preds, axis=1) == targs))
                return float(np.mean(preds == targs))
            if not no_pred:
                preds, targs = self.model.predict(X=X_test, y=y_test)
                self.save_preds(preds)

        except Exception as e:
            info = traceback.format_exc()
            print(f"Cleaning up {self.logdir} due to evaluation failure...")
            rmtree(self.logdir)
            print(f"Removed {self.logdir}")
            raise RuntimeError(f"Could not fit model:\n{info}") from e

    def save_preds(self, preds: ndarray) -> None:
        outfile = self.preds_dir / "preds.npz"
        np.savez_compressed(outfile, preds=preds)
        self.ckpt_file.touch(exist_ok=False)

    def load_preds(self) -> ndarray:
        outfile = self.preds_dir / "preds.npz"
        return np.load(outfile)["preds"]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Evaluator):
            return False
        if not (self.hparams == other.hparams):
            return False
        d1 = {**self.__dict__}
        d2 = {**other.__dict__}
        d1.pop("hparams")
        d2.pop("hparams")
        try:
            return d1 == d2
        except Exception as e:
            raise ValueError(missing_keys(d1, d2)) from e

    def __str__(self) -> str:
        fmt = ["Evaluator("]
        for key, val in self.__dict__.items():
            fmt.append(f"{key}={val}")
        fmt.append(")")
        return "\n".join(fmt)

    __repr__ = __str__


class Tuner(Evaluator):
    def __init__(
        self,
        dataset_name: DatasetName,
        classifier_kind: ClassifierKind,
        repeat: int,
        run: int,
        hparams: Hparams,
        dimension_reduction: Percentage | None,
        continuous_perturb: DataPerturbation | None,
        categorical_perturb: float | None,
        hparam_perturb: HparamPerturbation | None,
        train_downsample: Percentage | None,
        categorical_perturb_level: Literal["sample", "label"] = "label",
        debug: bool = False,
        _suppress_json: bool = False,
    ) -> None:
        super().__init__(
            dataset_name,
            classifier_kind,
            repeat,
            run,
            hparams,
            dimension_reduction,
            continuous_perturb,
            categorical_perturb,
            hparam_perturb,
            train_downsample,
            categorical_perturb_level,
            debug,
            _suppress_json,
        )

        self.logdir = self.get_logdir()

    def tune(self, no_pred: bool = False, return_test_acc: bool = False) -> float | None:
        if self.preds_dir.exists():
            with os.scandir(self.preds_dir) as files:
                if next(files, None) is not None:
                    raise FileExistsError(
                        "Impossible! Prediction data with same JOBID or hash "
                        f"already present in {self.res_dir}."
                    )
        ckpt_dir = self.ckpt_file.parent
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if len(ckpts) > 0:
                raise FileExistsError(
                    f"Checkpoint file {self.ckpt_file} exists, and suggests pred "
                    f"data for this evaluation repeat and run:\n{self}\n"
                    f"is already present in {self.logdir}."
                )
        try:
            ds = self.dataset
            # if self.classifer_kind in [ClassifierKind.MLP, ClassifierKind.LR]:
            #     raise NotImplementedError()
            X_train, y_train, X_test, y_test = ds.get_monte_carlo_splits(
                train_downsample=self.train_downsample,
                cont_perturb=self.continuous_perturb,
                cat_perturb_prob=self.categorical_perturb,
                cat_perturb_level=self.categorical_perturb_level,
                reduction=self.dimension_reduction,
                repeat=self.repeat,
                run=self.run,
            )
            self.model.fit(X=X_train, y=y_train)
            if return_test_acc:
                preds, targs = self.model.predict(X=X_test, y=y_test)
                if preds.ndim == 2:
                    return float(np.mean(np.argmax(preds, axis=1) == targs))
                return float(np.mean(preds == targs))
            if not no_pred:
                preds, targs = self.model.predict(X=X_test, y=y_test)
                self.save_preds(preds)

        except Exception as e:
            info = traceback.format_exc()
            print(f"Cleaning up {self.logdir} due to evaluation failure...")
            rmtree(self.logdir)
            print(f"Removed {self.logdir}")
            raise RuntimeError(f"Could not fit model:\n{info}") from e

    def get_logdir(self) -> Path:
        c = self.classifer_kind.value
        d = self.dataset_name.value
        dim = self.dimension_reduction
        red = "full" if dim is None else f"reduce={dim}"

        rep = f"rep={self.repeat:03d}"
        run = f"run={self.run:03d}"
        jid = os.environ.get("SLURM_JOB_ID")
        aid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if jid is None:
            slurm_id = None
        elif aid is not None:
            slurm_id = f"{jid}_{aid}"
        else:
            slurm_id = jid

        ts = strftime("%b-%d--%H-%M-%S")
        hsh = urlsafe_b64encode(uuid4().bytes).decode()
        uid = f"{ts}__{hsh}" if slurm_id is None else f"{slurm_id}__{ts}__{hsh}"
        root = DEBUG_LOGS if self.debug else LOGS
        return ensure_dir(root / f"tuning/{c}/{d}/{red}/{rep}/{run}/{uid}")

    def to_json(self, root: Path) -> None:
        root.mkdir(exist_ok=True, parents=True)
        hps = root / "hparams"
        out = root / "tuner.json"

        self.hparams.to_json(hps)
        with open(out, "w") as fp:
            json.dump(
                {
                    "dataset_name": self.dataset_name.value,
                    "classifier_kind": self.classifer_kind.value,
                    "dimension_reduction": self.dimension_reduction,
                    "continuous_perturb": value_or_none(self.continuous_perturb),
                    "categorical_perturb": self.categorical_perturb,
                    "hparam_perturb": value_or_none(self.hparam_perturb),
                    "train_downsample": self.train_downsample,
                    "categorical_perturb_level": self.categorical_perturb_level,
                    "repeat": self.repeat,
                    "run": self.run,
                    "debug": self.debug,
                },
                fp,
                indent=2,
            )

    @classmethod
    def from_json(cls: Type[Tuner], root: Path) -> Evaluator:
        root.mkdir(exist_ok=True, parents=True)
        hps = root / "hparams"
        out = root / "tuner.json"
        hparams = Hparams.from_json(hps)
        with open(out, "r") as fp:
            d = Namespace(**json.load(fp))

        c_perturb = to_enum_or_none(DataPerturbation, d.continuous_perturb)
        h_perturb = to_enum_or_none(HparamPerturbation, d.hparam_perturb)

        new = cls(
            dataset_name=DatasetName(d.dataset_name),
            classifier_kind=ClassifierKind(d.classifier_kind),
            hparams=hparams,
            dimension_reduction=d.dimension_reduction,
            continuous_perturb=c_perturb,
            categorical_perturb=d.categorical_perturb,
            hparam_perturb=h_perturb,
            train_downsample=d.train_downsample,
            categorical_perturb_level=d.categorical_perturb_level,
            repeat=d.repeat,
            run=d.run,
            debug=d.debug,
            _suppress_json=True,
        )
        new.logdir = root
        return new

    def __str__(self) -> str:
        fmt = ["Tuner("]
        for key, val in self.__dict__.items():
            fmt.append(f"{key}={val}")
        fmt.append(")")
        return "\n".join(fmt)
