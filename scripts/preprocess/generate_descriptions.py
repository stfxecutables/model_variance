from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.errors import PerformanceWarning
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.dataset import Dataset, DatasetName, reduce_categoricals
from src.enumerables import RuntimeClass


def describe_data(ds: Dataset) -> DataFrame:
    return ds.summary()


def describe_failsafe(ds: Dataset) -> Optional[DataFrame]:
    try:
        return describe_data(ds)
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e} for dataset {ds.name.name}")
        print(ds)
        return None


def make_description_table() -> None:
    datasets = [Dataset(dsname) for dsname in DatasetName]
    results = process_map(describe_failsafe, datasets, desc="Summarizing data")
    dfs = [r for r in results if r is not None]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    print(df.sort_values(by="n_onehot").to_markdown(tablefmt="simple"))


if __name__ == "__main__":
    filterwarnings("ignore", category=PerformanceWarning)
    make_description_table()
