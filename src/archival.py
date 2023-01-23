from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
import sys
import tarfile
from pathlib import Path
from tarfile import ExFileObject, ReadError, TarFile, TarInfo
from typing import Any

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from src.hparams.hparams import Hparams


def is_bad_tar(containing_tar: TarFile, info: TarInfo) -> bool:
    inner_fo: ExFileObject
    with containing_tar.extractfile(info) as inner_fo:  # type: ignore
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
    with inner_tar.extractfile(info) as exfile:  # type: ignore
        data = json.load(exfile)
    return data


def read_tar_npz(inner_tar: TarFile, name: str) -> dict[str, Any]:
    info: TarInfo = inner_tar.getmember(name)
    exfile: ExFileObject
    with inner_tar.extractfile(info) as exfile:  # type: ignore
        data = np.load(exfile)  # type: ignore
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
    missings: int = 0

    with open(tarpath, "rb") as fp:
        gz_archive: TarFile
        with tarfile.open(fileobj=fp, mode="r") as gz_archive:

            infos: list[TarInfo] = [*gz_archive.getmembers()]  # [:1000]
            for info in tqdm(infos):

                inner_archive: ExFileObject
                with gz_archive.extractfile(info) as inner_archive:  # type: ignore

                    # loops here should be cheap / acceptable relative to loops
                    # over full gz archive
                    inner_contents: TarFile
                    try:
                        with tarfile.open(
                            fileobj=inner_archive, mode="r"
                        ) as inner_contents:
                            fnames: list[str] = inner_contents.getnames()
                            evnames = list(
                                filter(lambda f: "evaluator.json" in f, fnames)
                            )
                            if len(evnames) == 0:
                                missings += 1
                                continue
                            evname = evnames[0]
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

    # get rid of useless `object` dtypes
    df = DataFrame(ev_dicts)
    df["dimension_reduction"] = df["dimension_reduction"].astype(float)
    df["categorical_perturb"] = df["categorical_perturb"].astype(float)
    df["train_downsample"] = df["train_downsample"].astype(float)
    for column in df.columns:
        col = df[column]
        if col.dtype == "object":
            df[column] = col.apply(str).astype("string")
    df["continuous_perturb"].fillna(value="None", inplace=True)
    df["hparam_perturb"].fillna(value="None", inplace=True)

    print(f"{missings} archives were missing key data.")

    return df, all_hps, all_preds, all_targs, read_fails
