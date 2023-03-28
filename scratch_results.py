from __future__ import annotations

import sys
from pathlib import Path

from src.results import Results

from scripts.testing.find_bad_tars import find_bad_tars

ROOT = Path(__file__).resolve().parent
PRELIM_DIR = ROOT / "debug_logs/prelim"
PRELIM = ROOT / "debug_logs/prelim.tar.gz"


if __name__ == "__main__":
    # bads = find_bad_tars(PRELIM)
    # if len(bads) > 0:
    #     for bad in bads:
    #         print(bad)
    # sys.exit()
    # results = Results.from_tar_gz(ROOT / "preliminary.tar", save_test=True)
    # results = Results.from_tar_gz(PRELIM)
    # results = Results.from_tar_gzs(PRELIM_DIR, cache=True)
    results = Results.from_cached(root=PRELIM_DIR)
    res = results.select()
    print("")
