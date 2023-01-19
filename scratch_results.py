from __future__ import annotations

import json
import pickle
import sys
import tarfile
from pathlib import Path
from tarfile import ExFileObject, ReadError, TarFile, TarInfo
from typing import Any, List, Optional, Sequence, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm
from typing_extensions import Literal

from src.archival import (
    find_bad_tars,
    is_bad_tar,
    parse_tar_gz,
    read_tar_json,
    read_tar_npz,
)
from src.constants import TESTING_TEMP
from src.evaluator import Evaluator
from src.hparams.hparams import Hparams

ROOT = Path(__file__).resolve().parent
TEST_TAR = TESTING_TEMP / "test_logs.tar"
TEST_EVALS_DF = TESTING_TEMP / "evals_test.parquet"
TEST_HPS_PICKLE = TESTING_TEMP / "hps_test.pickle"
TEST_PREDS = TESTING_TEMP / "test_preds.npz"
TEST_TARGS = TESTING_TEMP / "test_targs.npz"


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
        self.hps: List[Hparams] = hps
        self.preds: list[ndarray] = preds
        self.targs: list[ndarray] = targs

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
        with np.load(TEST_PREDS) as data:
            N = len(data)
            all_preds = [data[str(i)] for i in tqdm(range(N))]
        print("Loading targs")
        with np.load(TEST_TARGS) as data:
            N = len(data)
            all_targs = [data[str(i)] for i in tqdm(range(N))]

        return cls(evaluators=evals, hps=all_hps, preds=all_preds, targs=all_targs)


if __name__ == "__main__":
    # bads = find_bad_tars(TEST_TAR)
    # if len(bads) > 0:
    #     for bad in bads:
    #         print(bad)
    # sys.exit()
    results = Results.from_tar_gz(ROOT / "preliminary.tar", save_test=True)
    # results = Results.from_tar_gz(TEST_TAR)
    # results = Results.from_test_cached()
