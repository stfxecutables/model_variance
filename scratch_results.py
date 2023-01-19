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
