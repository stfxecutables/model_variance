from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import tarfile
from argparse import ArgumentParser
from pathlib import Path
from tarfile import ReadError, TarFile
from typing import List, Union

from tqdm.contrib.concurrent import process_map


def is_bad_tar(tarpath: Path) -> Union[Path, None]:
    with open(tarpath, "rb") as fp:
        gz_archive: TarFile
        try:
            with tarfile.open(fileobj=fp, mode="r") as gz_archive:
                ...
        except ReadError:
            return tarpath
    return None


def find_bad_tars(root: Path) -> List[Path]:
    tars = sorted(root.rglob("*.tar.gz"))
    paths = process_map(is_bad_tar, tars, desc="Checking archives", chunksize=10)
    paths = [p for p in paths if p is not None]
    if len(paths) > 0:
        print("Got bad archives:")
        for path in paths:
            print(path)
    return paths


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory from which to use Path.rglob('*.tar.gz') on.",
    )
    args = parser.parse_args()
    root = args.root
    find_bad_tars(root)
