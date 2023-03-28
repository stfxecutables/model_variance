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
from typing import Any, Dict, List, Optional, Tuple, cast

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
            infos: List[TarInfo] = [*outer_tar.getmembers()]
            for info in tqdm(infos):
                if is_bad_tar(outer_tar, info):
                    bads.append(info.name)
    return bads


def read_tar_json(inner_tar: TarFile, name: str) -> Dict[str, Any]:
    info: TarInfo = inner_tar.getmember(name)
    exfile: ExFileObject
    with inner_tar.extractfile(info) as exfile:  # type: ignore
        data = json.load(exfile)
    return cast(Dict[str, Any], data)


def read_tar_hparams(inner_tar: TarFile, name: str) -> Hparams:
    info: TarInfo = inner_tar.getmember(name)
    exfile: ExFileObject
    with inner_tar.extractfile(info) as exfile:  # type: ignore
        data: List[Dict[str, Any]] = json.load(exfile)
    return Hparams.from_dicts(data)


def read_tar_npz(inner_tar: TarFile, name: str) -> Dict[str, Any]:
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
) -> tuple[DataFrame, List[Hparams], List[ndarray], List[ndarray], int]:
    ev_dicts: List[Dict[str, Any]] = []
    all_hps: List[Hparams] = []
    all_preds: List[ndarray] = []
    all_targs: List[ndarray] = []
    read_fails: int = 0
    missings: int = 0

    with open(tarpath, "rb") as fp:
        gz_archive: TarFile
        with tarfile.open(fileobj=fp, mode="r") as gz_archive:

            infos: List[TarInfo] = [*gz_archive.getmembers()]  # [:1000]
            for info in tqdm(infos):
                if ".tar.gz" not in info.name:
                    continue

                inner_fp: ExFileObject = gz_archive.extractfile(info)
                inner_contents: TarFile
                try:
                    with tarfile.open(fileobj=inner_fp, mode="r") as inner_contents:
                        fnames: List[str] = inner_contents.getnames()
                        evnames = list(filter(lambda f: "evaluator.json" in f, fnames))
                        if len(evnames) == 0:
                            missings += 1
                            continue
                        evname = evnames[0]
                        hpfile = list(
                            filter(lambda f: "hparam" in f and ".json" in f, fnames)
                        )[0]
                        pnames = list(filter(lambda f: ".npz" in f, fnames))

                        ev = read_tar_json(inner_tar=inner_contents, name=evname)
                        hp = read_tar_hparams(inner_tar=inner_contents, name=hpfile)

                        preds, targs = sorted(
                            [read_tar_npz(inner_contents, p) for p in pnames],
                            key=lambda d: list(d.keys())[0],
                        )

                        ev_dicts.append(ev)  # type: ignore
                        all_hps.append(hp)
                        all_preds.append(preds["preds"])
                        all_targs.append(targs["targs"])
                except ReadError:
                    read_fails += 1
                finally:
                    inner_fp.close()

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


def unarchive_tar_gz(
    tarpath: Path,
) -> Optional[Tuple[DataFrame, Hparams, ndarray, ndarray]]:
    with open(tarpath, "rb") as fp:
        gz_archive: TarFile
        hp: Optional[Hparams] = None
        preds: Optional[ndarray] = None
        targs: Optional[ndarray] = None
        ev_dict: Optional[Dict[str, Any]] = None
        try:
            with tarfile.open(fileobj=fp, mode="r") as gz_archive:

                infos: List[TarInfo] = [*gz_archive.getmembers()]  # [:1000]
                for info in infos:
                    if info.isdir():
                        continue
                    fname = Path(info.name).name
                    if fname == "all_hparams.json":
                        hp = read_tar_hparams(inner_tar=gz_archive, name=info.name)
                    elif fname == "evaluator.json":
                        ev_dict = read_tar_json(inner_tar=gz_archive, name=info.name)
                    elif fname == "preds.npz":
                        preds = read_tar_npz(inner_tar=gz_archive, name=info.name)[
                            "preds"
                        ]
                    elif fname == "targs.npz":
                        targs = read_tar_npz(inner_tar=gz_archive, name=info.name)[
                            "targs"
                        ]
                    else:
                        continue

        except ReadError:
            return None

    if hp is None or preds is None or targs is None or ev_dict is None:
        return None

    # get rid of useless `object` dtypes
    df = DataFrame(ev_dict, index=[0])
    df["dimension_reduction"] = df["dimension_reduction"].astype("string")
    df["categorical_perturb"] = df["categorical_perturb"].astype(float)
    df["train_downsample"] = df["train_downsample"].astype(float)
    for column in df.columns:
        col = df[column]
        if col.dtype == "object":
            df[column] = col.apply(str).astype("string")
    df["continuous_perturb"].fillna(value="None", inplace=True)
    df["hparam_perturb"].fillna(value="None", inplace=True)

    return df, hp, preds, targs
