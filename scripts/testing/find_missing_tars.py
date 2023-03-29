from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import shutil
import sys
import tarfile
import traceback
from argparse import ArgumentParser
from pathlib import Path
from tarfile import ReadError, TarFile
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.constants import CKPTS
from src.parallelize import joblib_map


def is_bad_tar(tarpath: Path) -> bool:
    with open(tarpath, "rb") as fp:
        gz_archive: TarFile
        try:
            with tarfile.open(fileobj=fp, mode="r") as gz_archive:  # noqa
                infos = gz_archive.getmembers()
            if len(infos) < 5:
                return True
        except ReadError:
            return True
        except UnicodeDecodeError:
            return True
        except Exception:
            traceback.print_exc()
            return True
    return False


def find_bad_tars(root: Path) -> List[Path]:
    tars = sorted(root.rglob("*.tar.gz"))
    paths = process_map(is_bad_tar, tars, desc="Checking archives", chunksize=10)
    paths = [p for p in paths if p is not None]
    if len(paths) > 0:
        print("Got bad archives:")
        for path in paths:
            print(path)
    return paths


def is_bugged_ckpt(ckpt: Path) -> Optional[Path]:
    try:
        with open(ckpt, "r") as fp:
            tarfile = Path(f"{fp.readline()[:-1]}.tar.gz").resolve()  # trailing newline
        if not tarfile.exists():
            return tarfile
        if is_bad_tar(tarfile):
            return tarfile
        return None
    except IOError:
        traceback.print_exc()
        return ckpt


def print_bugged_ckpts(root: Path = CKPTS) -> List[Path]:
    ckptfiles = root.rglob("*.ckpt")
    ckpts = [ckpt for ckpt in tqdm(ckptfiles, desc="Globbing .ckpts")]
    results = joblib_map(is_bugged_ckpt, ckpts, desc="Checking for missing .tar.gz files")
    missings = [r for r in results if r is not None]
    for missing in missings:
        print(missing)
    return missings


def dedup_tars(root: Path, delete: bool = False) -> None:
    tars = list(root.rglob("*.tar.gz"))
    parents, counts = np.unique([tar.parent for tar in tars], return_counts=True)
    unq_parents = parents[counts > 1]
    duplicates = []
    dirs = []
    corrupts = []
    to_delete = []
    for unq_parent in unq_parents:
        files = list(unq_parent.rglob("*"))
        file: Path
        for file in files:
            if "".join(file.suffixes) != ".tar.gz":
                if file.is_dir():
                    try:
                        file.rmdir()
                    except Exception:
                        dirs.append(file)
                else:
                    file.unlink()
            else:
                duplicates.append(file)
        targzs = filter(lambda f: "".join(f.suffixes) == ".tar.gz", files)
        good_tars: List[Path] = []
        for targz in targzs:
            if is_bad_tar(targz):
                corrupts.append(targz)
            else:
                good_tars.append(targz)
        # sort so first item is most recent
        goods = sorted(good_tars, key=lambda p: p.stat().st_mtime, reverse=True)
        to_delete.extend(goods[1:])

    # print("Directories with duplicate archives:")
    # for dup in duplicates:
    #     print(dup)
    d: Path
    print(f"Total archives found: {len(tars)}")
    print(f"Leftover sub-directories: {len(dirs)}")
    for d in dirs:
        print(d)
    print(f"Corrupt archives: {len(corrupts)}")
    for c in corrupts:
        print(c)
    print(f"Duplicate archives to delete: {len(to_delete)}")
    if len(to_delete) <= 10:
        for d in to_delete:
            print(d)
    else:
        for d in to_delete[:5]:
            print(d)
        print("...")
        for d in to_delete[-5:]:
            print(d)
    if delete:
        for d in to_delete:
            d.unlink()


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument(
    #     "--root",
    #     type=Path,
    #     required=True,
    #     help="Directory from which to use Path.rglob('*.ckpt') on.",
    # )
    # parser.add_argument(
    #     "--tar-root",
    #     type=Path,
    #     required=True,
    #     help="Directory to first dedup",
    # )
    # args = parser.parse_args()
    # root = args.root
    # print_bugged_ckpts(root)

    PRELIM = ROOT / "debug_logs/prelim"
    dedup_tars(PRELIM, delete=False)
