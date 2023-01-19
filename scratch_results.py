from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from tarfile import ExFileObject, TarFile, TarInfo
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

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
) -> tuple[list[dict], list[dict], list[ndarray], list[ndarray]]:
    ev_dicts: list[dict[str, Any]] = []
    all_hps: list[list[dict[str, Any]]] = []
    all_preds: list[ndarray] = []
    all_targs: list[ndarray] = []

    with open(tarpath, "rb") as fp:
        gz_archive: TarFile
        with tarfile.open(fileobj=fp, mode="r") as gz_archive:

            infos: list[TarInfo] = [*gz_archive.getmembers()]
            for info in tqdm(infos):

                inner_archive: ExFileObject
                with gz_archive.extractfile(info) as inner_archive:

                    # loops here should be cheap / acceptable relative to loops
                    # over full gz archive
                    inner_contents: TarFile
                    with tarfile.open(fileobj=inner_archive, mode="r") as inner_contents:
                        fnames: list[str] = inner_contents.getnames()
                        evname = list(filter(lambda f: "evaluator.json" in f, fnames))[0]
                        hpnames = list(
                            filter(lambda f: "hparam" in f and ".json" in f, fnames)
                        )
                        pnames = list(filter(lambda f: ".npz" in f, fnames))

                        ev = read_tar_json(inner_tar=inner_contents, name=evname)
                        hps = [
                            read_tar_json(inner_tar=inner_contents, name=hp)
                            for hp in hpnames
                        ]
                        preds, targs = sorted(
                            [read_tar_npz(inner_contents, p) for p in pnames],
                            key=lambda d: list(d.keys())[0],
                        )

                        ev_dicts.append(ev)
                        all_hps.append(hps)
                        all_preds.append(preds["preds"])
                        all_targs.append(targs["targs"])
    return ev_dicts, all_hps, all_preds, all_targs


class Results:
    def __init__(self) -> None:
        # indexed by repeat, run?
        self.evaluators: list[Evaluator]
        self.preds: list[ndarray]
        self.targs: list[ndarray]

    @classmethod
    def from_tar_gz(cls, targz: Path) -> Results:
        archive_files = [
            "evaluator.json",
            # "average.json",
            # "penalty.json",
            # "alpha.json",
            # "eta0.json",
            # "l1_ratio.json",
            # "learning_rate.json",
            # "loss.json",
            "preds.npz",
            "targs.npz",
        ]
        archive_dirs = [
            "preds",
            "hparams",
            "fixed",
            "continuous",
            "categorical",
        ]
        ev_dicts, all_hps, all_preds, all_targs = parse_tar_gz(targz)
        print()


if __name__ == "__main__":
    results = Results.from_tar_gz(TEST_TAR)
