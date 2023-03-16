from __future__ import annotations

from pathlib import Path

from src.results import Results

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    # bads = find_bad_tars(TEST_TAR)
    # if len(bads) > 0:
    #     for bad in bads:
    #         print(bad)
    # sys.exit()
    # results = Results.from_tar_gz(ROOT / "preliminary.tar", save_test=True)
    # results = Results.from_tar_gz(TEST_TAR)
    results = Results.from_test_cached()
    res = results.select()
    print("")
