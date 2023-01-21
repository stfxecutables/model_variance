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
from shutil import make_archive, rmtree
from time import sleep, strftime
from typing import Any, Literal, Type, TypeVar, overload
from uuid import uuid4

import numpy as np
from numpy import ndarray

from src.constants import CKPTS, DEBUG_LOGS, LOGS, ensure_dir
from src.dataset import Dataset
from src.enumerables import (
    CatPerturbLevel,
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
    get_index,
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


def ckpt_file(
    dataset_name: DatasetName,
    classifier_kind: ClassifierKind,
    repeat: int,
    run: int,
    dimension_reduction: Percentage | None,
    continuous_perturb: DataPerturbation | None,
    categorical_perturb: float | None,
    hparam_perturb: HparamPerturbation | None,
    train_downsample: Percentage | None,
    categorical_perturb_level: CatPerturbLevel,
    label: str = "debug",
    **kwargs: Any,  # to ignore
) -> Path:
    cid = "_".join(
        [
            f"{get_index(dataset_name)}",
            f"{get_index(classifier_kind)}",
            f"{get_index(repeat)}",
            f"{get_index(run)}",
            f"{get_index(dimension_reduction)}",
            f"{get_index(continuous_perturb)}",
            f"{categorical_perturb}".replace("None", "0.0").replace(".", "-"),
            f"{get_index(hparam_perturb)}",
            f"{get_index(train_downsample)}",
            f"{get_index(categorical_perturb_level)}",
        ]
    )
    outdir = CKPTS / label
    ensure_dir(outdir)
    return outdir / f"{cid}.ckpt"


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
        categorical_perturb_level: CatPerturbLevel = CatPerturbLevel.Label,
        label: str | None = None,
        debug: bool = False,
        _suppress_json: bool = False,
    ) -> None:
        self.dataset_name: DatasetName = dataset_name
        self.dataset_: Dataset | None = None
        self.classifier_kind: ClassifierKind = classifier_kind
        self.repeat: int = repeat
        self.run: int = run
        self.hparams: Hparams = hparams
        self.dimension_reduction: Percentage | None = dimension_reduction
        self.continuous_perturb: DataPerturbation | None = continuous_perturb
        self.categorical_perturb: float | None = categorical_perturb
        self.hparam_perturb: HparamPerturbation | None = hparam_perturb
        self.train_downsample: Percentage | None = train_downsample
        self._model: ClassifierModel | None = None
        self.categorical_perturb_level: CatPerturbLevel = categorical_perturb_level
        self.debug = debug
        self._label = label
        if self._label is None:
            self.label = "debug" if self.debug else "eval"
        else:
            self.label = self._label

        self.logdir = self.setup_logdir()
        if not _suppress_json:
            self.to_json(self.logdir)
        if not debug:
            print(f"Evaluator results will be logged to {self.logdir}")

    @property
    def model(self) -> ClassifierModel:
        if self._model is not None:
            return self._model
        kind = self.classifier_kind
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
            raise ValueError(f"Unknown model kind: {self.classifier_kind}")
        return self._model

    def get_id(self) -> str:
        ckpt = self.ckpt_file
        return ckpt.stem.replace(".ckpt", "")

    def setup_logdir(self) -> Path:
        root = DEBUG_LOGS if self.debug else LOGS
        arg_id = self.get_id()

        c = self.classifier_kind.value
        d = self.dataset_name.value
        dim = self.dimension_reduction
        red = "full" if dim is None else f"reduce={dim}"

        rep = f"rep={self.repeat:03d}"
        run = f"run={self.run:03d}"

        logdir = ensure_dir(root / arg_id)

        ts = strftime("%b-%d--%H-%M-%S")
        hsh = urlsafe_b64encode(uuid4().bytes).decode()
        uid = f"{ts}__{hsh}"
        # we put arg_id near very end to make deletions of failed / corrupt jobs easy
        logdir = ensure_dir(
            root / f"{self.label}/{d}/{c}/{red}/{rep}/{run}/{arg_id}/{uid}"
        )

        return logdir

    @property
    def res_dir(self) -> Path:
        """This a dynamic property so it matches `self.logdir` on deserialization"""
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
        return ckpt_file(
            dataset_name=self.dataset_name,
            classifier_kind=self.classifier_kind,
            repeat=self.repeat,
            run=self.run,
            dimension_reduction=self.dimension_reduction,
            continuous_perturb=self.continuous_perturb,
            categorical_perturb=self.categorical_perturb,
            hparam_perturb=self.hparam_perturb,
            train_downsample=self.train_downsample,
            categorical_perturb_level=self.categorical_perturb_level,
            label=self.label,
        )

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
                    "classifier_kind": self.classifier_kind.value,
                    "dimension_reduction": self.dimension_reduction,
                    "continuous_perturb": value_or_none(self.continuous_perturb),
                    "categorical_perturb": self.categorical_perturb,
                    "hparam_perturb": value_or_none(self.hparam_perturb),
                    "train_downsample": self.train_downsample,
                    "categorical_perturb_level": self.categorical_perturb_level.value,
                    "repeat": self.repeat,
                    "run": self.run,
                    "debug": self.debug,
                    "label": self._label,
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
        cat_perturb = CatPerturbLevel(d.categorical_perturb_level)

        new = cls(
            dataset_name=DatasetName(d.dataset_name),
            classifier_kind=ClassifierKind(d.classifier_kind),
            hparams=hparams,
            dimension_reduction=d.dimension_reduction,
            continuous_perturb=c_perturb,
            categorical_perturb=d.categorical_perturb,
            hparam_perturb=h_perturb,
            train_downsample=d.train_downsample,
            categorical_perturb_level=cat_perturb,
            repeat=d.repeat,
            run=d.run,
            label=d.label,
            debug=d.debug,
            _suppress_json=True,
        )
        new.logdir = root
        return new

    def cleanup(self, remove_ckpt: bool = True, silent: bool = True) -> None:
        try:
            needs_cleanup = self.logdir.exists() or (
                self.ckpt_file.exists() and remove_ckpt
            )
            if self.logdir.exists():
                if not silent:
                    print(f"Cleaning up {self.logdir}...")
                rmtree(self.logdir)
            if self.ckpt_file.exists() and remove_ckpt:
                self.ckpt_file.unlink(missing_ok=True)
            if needs_cleanup:
                if not silent:
                    print(f"Removed {self.logdir} and checkpoint file {self.ckpt_file}")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to cleanup run data. Data may remain at either "
                f"{self.logdir} or {self.ckpt_file}. "
            ) from e

    @overload
    def evaluate(
        self,
        no_pred: bool = False,
        return_test_acc: Literal[True] = True,
        skip_done: bool = False,
    ) -> float:
        ...

    @overload
    def evaluate(
        self,
        no_pred: Literal[True] = True,
        return_test_acc: bool = True,
        skip_done: bool = False,
    ) -> None:
        ...

    @overload
    def evaluate(
        self,
        no_pred: bool = False,
        return_test_acc: Literal[False] = False,
        skip_done: bool = False,
    ) -> None:
        ...

    def evaluate(
        self, no_pred: bool = False, return_test_acc: bool = False, skip_done: bool = True
    ) -> float | None:
        if not skip_done:
            self.ckpt_file.unlink(missing_ok=True)
        if skip_done and self.ckpt_file.exists():
            self.cleanup(remove_ckpt=False, silent=True)
            return None
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
            if no_pred:
                return None

            preds, targs = self.model.predict(X=X_test, y=y_test)
            self.save_preds(preds)
            self.save_targs(targs)
            self.archive()
            sleep(5)  # dunno, maybe this prevents sudden stops?
            with open(self.ckpt_file, "w+") as fp:
                fp.write(f"{self.logdir}\n")
            self.cleanup(remove_ckpt=False, silent=True)

            if return_test_acc:
                if preds.ndim == 2:
                    return float(np.mean(np.argmax(preds, axis=1) == targs))
                return float(np.mean(preds == targs))

        except Exception as e:
            info = traceback.format_exc()
            print(f"Cleaning up {self.logdir} due to evaluation failure...")
            self.cleanup(remove_ckpt=True)
            raise RuntimeError(f"Could not fit model:\n{info}") from e

    def archive(self) -> None:
        root = self.logdir
        make_archive(str(root), format="gztar", root_dir=root, base_dir=root)

    def save_preds(self, preds: ndarray) -> None:
        outfile = self.preds_dir / "preds.npz"
        np.savez_compressed(outfile, preds=preds)

    def save_targs(self, targs: ndarray) -> None:
        outfile = self.preds_dir / "targs.npz"
        np.savez_compressed(outfile, targets=targs)

    def load_preds(self) -> ndarray:
        outfile = self.preds_dir / "preds.npz"
        return np.load(outfile)["preds"]

    def _ensure_no_checkpoint(self) -> None:
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Evaluator):
            return False
        if not (self.hparams == other.hparams):
            return False
        d1 = {**self.__dict__}
        d2 = {**other.__dict__}
        ignored = ["logdir"]
        for ignore in ignored:
            d1.pop(ignore)
            d2.pop(ignore)
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
        categorical_perturb_level: CatPerturbLevel = CatPerturbLevel.Label,
        label: str | None = None,
        debug: bool = False,
        _suppress_json: bool = False,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            classifier_kind=classifier_kind,
            repeat=repeat,
            run=run,
            hparams=hparams,
            dimension_reduction=dimension_reduction,
            continuous_perturb=continuous_perturb,
            categorical_perturb=categorical_perturb,
            hparam_perturb=hparam_perturb,
            train_downsample=train_downsample,
            categorical_perturb_level=categorical_perturb_level,
            label=label,
            debug=debug,
            _suppress_json=True,
        )
        if self._label is None:
            self.label = "debug" if self.debug else "tune"
        else:
            self.label = self._label
        self.logdir = self.setup_logdir()
        if not _suppress_json:
            self.to_json(self.logdir)

    def tune(
        self, no_pred: bool = False, return_test_acc: bool = False, skip_done: bool = True
    ) -> float | None:
        try:
            if self.ckpt_file.exists() and skip_done:
                return None
            ds = self.dataset
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

    def setup_logdir(self) -> Path:
        arg_id = self.get_id()
        c = self.classifier_kind.value
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
        return ensure_dir(
            root / f"tuning/{self.label}/{c}/{d}/{red}/{rep}/{run}/{arg_id}/{uid}"
        )

    @property
    def ckpt_file(self) -> Path:
        """This should be a unique"""
        return ckpt_file(
            dataset_name=self.dataset_name,
            classifier_kind=self.classifier_kind,
            repeat=self.repeat,
            run=self.run,
            dimension_reduction=self.dimension_reduction,
            continuous_perturb=self.continuous_perturb,
            categorical_perturb=self.categorical_perturb,
            hparam_perturb=self.hparam_perturb,
            train_downsample=self.train_downsample,
            categorical_perturb_level=self.categorical_perturb_level,
            label=self.label,
        )

    def to_json(self, root: Path) -> None:
        root.mkdir(exist_ok=True, parents=True)
        hps = root / "hparams"
        out = root / "tuner.json"

        self.hparams.to_json(hps)
        with open(out, "w") as fp:
            json.dump(
                {
                    "dataset_name": self.dataset_name.value,
                    "classifier_kind": self.classifier_kind.value,
                    "dimension_reduction": self.dimension_reduction,
                    "continuous_perturb": value_or_none(self.continuous_perturb),
                    "categorical_perturb": self.categorical_perturb,
                    "hparam_perturb": value_or_none(self.hparam_perturb),
                    "train_downsample": self.train_downsample,
                    "categorical_perturb_level": self.categorical_perturb_level.value,
                    "repeat": self.repeat,
                    "label": self._label,
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
        cat_perturb = CatPerturbLevel(d.categorical_perturb_level)

        new = cls(
            dataset_name=DatasetName(d.dataset_name),
            classifier_kind=ClassifierKind(d.classifier_kind),
            hparams=hparams,
            dimension_reduction=d.dimension_reduction,
            continuous_perturb=c_perturb,
            categorical_perturb=d.categorical_perturb,
            hparam_perturb=h_perturb,
            train_downsample=d.train_downsample,
            categorical_perturb_level=cat_perturb,
            repeat=d.repeat,
            run=d.run,
            label=d.label,
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
