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

from src.evaluator import Evaluator
from src.hparams.hparams import Hparams

ROOT = Path(__file__).resolve().parent
TEST_TAR = ROOT / "test_logs.tar"
TEST_EVALS_DF = ROOT / "evals_test.parquet"
TEST_HPS_PICKLE = ROOT / "hps_test.pickle"
TEST_PREDS = ROOT / "test_preds.npz"
TEST_TARGS = ROOT / "test_targs.npz"


def is_bad_tar(containing_tar: TarFile, info: TarInfo) -> bool:
    inner_fo: ExFileObject
    with containing_tar.extractfile(info) as inner_fo:
        try:
            inner_tar = tarfile.open(fileobj=inner_fo, mode="r")
            inner_tar.close()
        except ReadError:
            return True
    return False


def find_bad_tars(targz_path: Path) -> Any:
    bads = []
    with open(targz_path, "rb") as fp:
        outer_tar: TarFile
        with tarfile.open(fileobj=fp, mode="r") as outer_tar:
            infos: list[TarInfo] = [*outer_tar.getmembers()]
            for info in tqdm(infos):
                if is_bad_tar(outer_tar, info):
                    bads.append(info.name)
    return bads


def read_tar_json(inner_tar: TarFile, name: str) -> dict[str, Any]:
    info: TarInfo = inner_tar.getmember(name)
    exfile: ExFileObject
    with inner_tar.extractfile(info) as exfile:
        data = json.load(exfile)
    return data


def read_tar_npz(inner_tar: TarFile, name: str) -> dict[str, Any]:
    info: TarInfo = inner_tar.getmember(name)
    exfile: ExFileObject
    with inner_tar.extractfile(info) as exfile:
        data = np.load(exfile)
        if "preds" in data:
            return {"preds": data["preds"]}
        elif "targets" in data:
            return {"targs": data["targets"]}
        else:
            raise ValueError("Unrecognized NumPy data in archive")


def parse_tar_gz(
    tarpath: Path,
) -> tuple[DataFrame, list[Hparams], list[ndarray], list[ndarray], int]:
    ev_dicts: list[dict[str, Any]] = []
    all_hps: list[Hparams] = []
    all_preds: list[ndarray] = []
    all_targs: list[ndarray] = []
    read_fails: int = 0

    with open(tarpath, "rb") as fp:
        gz_archive: TarFile
        with tarfile.open(fileobj=fp, mode="r") as gz_archive:

            infos: list[TarInfo] = [*gz_archive.getmembers()]  # [:1000]
            for info in tqdm(infos):

                inner_archive: ExFileObject
                with gz_archive.extractfile(info) as inner_archive:

                    # loops here should be cheap / acceptable relative to loops
                    # over full gz archive
                    inner_contents: TarFile
                    try:
                        with tarfile.open(
                            fileobj=inner_archive, mode="r"
                        ) as inner_contents:
                            fnames: list[str] = inner_contents.getnames()
                            evname = list(
                                filter(lambda f: "evaluator.json" in f, fnames)
                            )[0]
                            hpnames = list(
                                filter(lambda f: "hparam" in f and ".json" in f, fnames)
                            )
                            pnames = list(filter(lambda f: ".npz" in f, fnames))

                            ev = read_tar_json(inner_tar=inner_contents, name=evname)
                            hps = [
                                read_tar_json(inner_tar=inner_contents, name=hp)
                                for hp in hpnames
                            ]
                            hp = Hparams.from_dicts(hps)

                            preds, targs = sorted(
                                [read_tar_npz(inner_contents, p) for p in pnames],
                                key=lambda d: list(d.keys())[0],
                            )

                            ev_dicts.append(ev)
                            all_hps.append(hp)
                            all_preds.append(preds["preds"])
                            all_targs.append(targs["targs"])
                    except ReadError:
                        read_fails += 1
    return DataFrame(ev_dicts), all_hps, all_preds, all_targs, read_fails


class Results:
    def __init__(self, evaluators: DataFrame, hps: list[Hparams], preds: list[ndarray], targs: list[ndarray]) -> None:
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
            np.savez_compressed(
                TEST_PREDS, **{str(i): arr for i, arr in enumerate(all_preds)}
            )
            np.savez_compressed(
                TEST_TARGS, **{str(i): arr for i, arr in enumerate(all_targs)}
            )
        if read_fails > 0:
            print(f"Failed to read {read_fails} archives. Run `find_bad_tars.py` to find.")
        return cls(evaluators=evals, hps=all_hps, preds=all_preds, targs=all_targs)

    @classmethod
    def from_test_cached(cls: Type[Results]) -> Results:
        evals = pd.read_parquet(TEST_EVALS_DF)
        with open(TEST_HPS_PICKLE, "rb") as fp:
            all_hps = pickle.load(fp)
        with np.load(TEST_PREDS) as data:
            N = len(data)
            all_preds = [data[str(i)] for i in range(N)]
        with np.load(TEST_TARGS) as data:
            N = len(data)
            all_targs = [data[str(i)] for i in range(N)]

        return cls(evaluators=evals, hps=all_hps, preds=all_preds, targs=all_targs)



if __name__ == "__main__":
    # bads = find_bad_tars(TEST_TAR)
    # if len(bads) > 0:
    #     for bad in bads:
    #         print(bad)
    # sys.exit()
    # results = Results.from_tar_gz(TEST_TAR)
    results = Results.from_test_cached()
