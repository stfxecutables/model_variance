from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import pickle
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Type, TypeVar, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from src.archival import parse_tar_gz
from src.constants import TESTING_TEMP
from src.enumerables import (
    CatPerturbLevel,
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
)
from src.hparams.hparams import Hparams

TEST_TAR = TESTING_TEMP / "test_logs.tar"
TEST_EVALS_DF = TESTING_TEMP / "evals_test.parquet"
TEST_HPS_PICKLE = TESTING_TEMP / "hps_test.pickle"
TEST_PREDS = TESTING_TEMP / "test_preds.npz"
TEST_TARGS = TESTING_TEMP / "test_targs.npz"
TEST_PREDS_PICKLE = TESTING_TEMP / "test_preds.pickle"
TEST_TARGS_PICKLE = TESTING_TEMP / "test_targs.pickle"

All = Literal["all"]
Enums = Union[
    DatasetName,
    DataPerturbation,
    ClassifierKind,
    CatPerturbLevel,
    HparamPerturbation,
]


class Results:
    def __init__(
        self,
        evaluators: DataFrame,
        hps: list[Hparams],
        preds: list[ndarray],
        targs: list[ndarray],
    ) -> None:
        # indexed by repeat, run?
        self.evaluators: DataFrame = evaluators
        self.hps: list[Hparams] = hps
        self.preds: list[ndarray] = preds
        self.targs: list[ndarray] = targs

    def select(
        self,
        dsnames: list[DatasetName] | All = "all",
        classifier_kinds: list[ClassifierKind] | All = "all",
        reductions: list[Literal[25, 50, 75] | None] | All = "all",
        cont_perturb: list[DataPerturbation | None] | All = "all",
        cat_perturb: list[DataPerturbation | None] | All = "all",
        hp_perturb: list[HparamPerturbation | None] | All = "all",
        train_downsample: list[Literal[25, 50, 75] | None] | All = "all",
        cat_perturb_level: list[CatPerturbLevel] | All = "all",
    ) -> Results:
        def to_floats(vals: list[Any]) -> list[float]:
            ret = []
            for val in vals:
                ret.append(float("nan") if val is None else float(val))
            return ret

        def to_strs(vals: list[Any]) -> list[str]:
            ret = []
            for val in vals:
                ret.append(val.value if isinstance(val, Enum) else "None")
            return ret

        df = self.evaluators
        idx = df["debug"].isin([True, False])
        if dsnames != "all":
            idx &= df["dataset_name"].isin(to_strs(dsnames))
        if classifier_kinds != "all":
            idx &= df["classifier_kind"].isin(to_strs(classifier_kinds))
        if reductions != "all":
            idx &= df["dimension_reduction"].isin(to_floats(reductions))
        if cont_perturb != "all":
            idx &= df["continuous_perturb"].isin(to_strs(cont_perturb))
        if cat_perturb != "all":
            idx &= df["categorical_perturb"].isin(to_floats(cat_perturb))
        if hp_perturb != "all":
            idx &= df["hparam_perturb"].isin(to_strs(hp_perturb))
        if train_downsample != "all":
            idx &= df["train_downsample"].isin(to_floats(train_downsample))
        if cat_perturb_level != "all":
            idx &= df["categorical_perturb_level"].isin(to_strs(cat_perturb_level))

        return Results(
            evaluators=df.loc[idx].copy(),
            hps=np.array(self.hps, dtype=object, copy=False)[idx].tolist(),
            preds=np.array(self.preds, dtype=object, copy=False)[idx].tolist(),
            targs=np.array(self.targs, dtype=object, copy=False)[idx].tolist(),
        )

        ...

    @classmethod
    def from_tar_gz(cls: Type[Results], targz: Path, save_test: bool = False) -> Results:
        evals, all_hps, all_preds, all_targs, read_fails = parse_tar_gz(targz)
        if save_test:
            evals.to_parquet(TEST_EVALS_DF)
            with open(TEST_HPS_PICKLE, "wb") as fp:
                pickle.dump(all_hps, fp)
            np.savez(TEST_PREDS, **{str(i): arr for i, arr in enumerate(all_preds)})
            np.savez(TEST_TARGS, **{str(i): arr for i, arr in enumerate(all_targs)})
        if read_fails > 0:
            print(
                f"Failed to read {read_fails} archives. Run `find_bad_tars.py` to find."
            )
        return cls(evaluators=evals, hps=all_hps, preds=all_preds, targs=all_targs)

    @classmethod
    def from_test_cached(cls: Type[Results]) -> Results:
        print("Loading evals df")
        evals = pd.read_parquet(TEST_EVALS_DF)
        print("Loading hparams")
        with open(TEST_HPS_PICKLE, "rb") as fp:
            all_hps = pickle.load(fp)
        print("Loading preds")
        with open(TEST_PREDS_PICKLE, "rb") as fp:
            all_preds = pickle.load(fp)
        # with np.load(TEST_PREDS) as data:
        #     N = len(data)
        #     all_preds = [data[str(i)] for i in tqdm(range(N))]
        print("Loading targs")
        with open(TEST_TARGS_PICKLE, "rb") as fp:
            all_targs = pickle.load(fp)
        # with np.load(TEST_TARGS) as data:
        #     N = len(data)
        #     all_targs = [data[str(i)] for i in tqdm(range(N))]

        return cls(evaluators=evals, hps=all_hps, preds=all_preds, targs=all_targs)

    def __str__(self) -> str:
        return str(self.evaluators)

    __repr__ = __str__
