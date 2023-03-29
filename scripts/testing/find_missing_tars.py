from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import tarfile
import traceback
from argparse import ArgumentParser
from pathlib import Path
from tarfile import ReadError, TarFile
from typing import List, Optional, Union

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory from which to use Path.rglob('*.ckpt') on.",
    )
    args = parser.parse_args()
    root = args.root
    print_bugged_ckpts(root)
