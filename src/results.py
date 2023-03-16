from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import pickle
import sys
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any, List, Literal, Optional, Type, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm
from typing_extensions import Literal

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


@dataclass
class PredTarg:
    preds: ndarray
    targs: ndarray


class Results:
    ALL_COLS = [
        "dataset_name",
        "classifier_kind",
        "continuous_perturb",
        "categorical_perturb",
        "hparam_perturb",
        "train_downsample",
        "categorical_perturb_level",
        "repeat",
        "run",
    ]
    GRP_COLS = ALL_COLS[:-1]

    def __init__(
        self,
        evaluators: DataFrame,
        hps: List[Hparams],
        preds: List[ndarray],
        targs: List[ndarray],
    ) -> None:
        # indexed by repeat, run?
        self.evaluators: DataFrame = evaluators
        self.hps: List[Hparams] = hps
        self.preds: List[ndarray] = preds
        self.targs: List[ndarray] = targs

    def select(
        self,
        dsnames: Union[List[DatasetName], All] = "all",
        classifier_kinds: Union[List[ClassifierKind], All] = "all",
        reductions: Union[List[Optional[Literal[25, 50, 75]]], All] = "all",
        cont_perturb: Union[List[Optional[DataPerturbation]], All] = "all",
        cat_perturb: Union[List[Optional[DataPerturbation]], All] = "all",
        hp_perturb: Union[List[Optional[HparamPerturbation]], All] = "all",
        train_downsample: Union[List[Optional[Literal[25, 50, 75]]], All] = "all",
        cat_perturb_level: Union[List[CatPerturbLevel], All] = "all",
        repeats: Union[List[int], All] = "all",
        runs: Union[List[int], All] = "all",
    ) -> Results:
        def to_floats(vals: List[Any]) -> List[float]:
            ret = []
            for val in vals:
                ret.append(float("nan") if val is None else float(val))
            return ret

        def to_strs(vals: List[Any]) -> List[str]:
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
        if repeats != "all":
            idx &= df["repeat"].isin(repeats)
        if runs != "all":
            idx &= df["run"].isin(runs)

        idx_int = np.where(idx)[0].tolist()

        return Results(
            evaluators=df.loc[idx].copy(),
            # hps=np.array(self.hps, dtype=object, copy=False)[idx].tolist(),
            hps=[self.hps[i] for i in idx_int],
            preds=[self.preds[i] for i in idx_int],
            targs=[self.targs[i] for i in idx_int],
        )

    def repeat_dfs(self) -> tuple[List[int], List[DataFrame]]:
        df = self.evaluators.copy()
        df.categorical_perturb.fillna(0.0, inplace=True)
        df.train_downsample.fillna(100, inplace=True)

        # reduce df to non-constant columns
        cols = df.columns.to_list()
        const_cols = [c for c in cols if len(df[c].unique()) == 1]
        for col in const_cols:
            cols.pop(cols.index(col))
        cols.pop(cols.index("run"))

        dfs = []
        reps = []
        for _, sub_df in df.groupby(cols):
            rep = int(sub_df["repeat"].iloc[0])
            dfs.append(sub_df.sort_values(by="run", ascending=True).drop(columns="run"))
            # dfs.append(sub_df)
            reps.append(rep)
        return reps, dfs

    def repeat_pairs(self, repeat: int) -> List[tuple[int, int]]:
        """Return indices of pairs matching `repeat`"""
        df = self.evaluators["repeat"]
        idx = df[df.isin([repeat])].index.to_list()
        combs = list(combinations(idx, 2))
        return combs

    def repeat_preds_targs(self, repeat: int) -> List[tuple[PredTarg, PredTarg]]:
        idx = self.repeat_pairs(repeat)
        pairs = []
        for i1, i2 in idx:
            pairs.append(
                (
                    PredTarg(preds=self.preds[i1], targs=self.targs[i1]),
                    PredTarg(preds=self.preds[i2], targs=self.targs[i2]),
                )
            )
        return pairs

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
        if TEST_PREDS_PICKLE.exists():
            with open(TEST_PREDS_PICKLE, "rb") as fp:
                all_preds = pickle.load(fp)
        else:
            with np.load(TEST_PREDS) as data:
                N = len(data)
                all_preds = [data[str(i)] for i in tqdm(range(N))]
            with open(TEST_PREDS_PICKLE, "wb") as fp:
                pickle.dump(all_preds, fp)

        print("Loading targs")
        if TEST_TARGS_PICKLE.exists():
            with open(TEST_TARGS_PICKLE, "rb") as fp:
                all_targs = pickle.load(fp)
        else:
            with np.load(TEST_TARGS) as data:
                N = len(data)
                all_targs = [data[str(i)] for i in tqdm(range(N))]
            with open(TEST_TARGS_PICKLE, "wb") as fp:
                pickle.dump(all_targs, fp)

        return cls(evaluators=evals, hps=all_hps, preds=all_preds, targs=all_targs)

    def __str__(self) -> str:
        return str(self.evaluators)

    __repr__ = __str__
