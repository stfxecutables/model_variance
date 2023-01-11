from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src.constants import CC_RESULTS


def set_long_print() -> None:
    pd.options.display.max_rows = 5000
    pd.options.display.max_info_rows = 5000
    pd.options.display.max_columns = 1000
    pd.options.display.max_info_columns = 1000
    pd.options.display.large_repr = "truncate"
    pd.options.display.expand_frame_repr = True
    pd.options.display.width = 200


def to_readable(duration_s: float) -> str:
    if duration_s <= 120:
        return f"{np.round(duration_s, 1):03.1f} sec"
    mins = duration_s / 60
    if mins <= 120:
        return f"{np.round(mins, 1):03.1f} min"
    hrs = mins / 60
    return f"{np.round(hrs, 2):03.2f} hrs"


if __name__ == "__main__":
    set_long_print()

    times = CC_RESULTS / "niagara/results/runtimes"
    jsons = sorted(times.rglob("*runtimes.json"))
    dfs = []
    for js in jsons:
        df = pd.read_json(js)
        df["classifier"] = js.name[: js.name.find("_")]
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    summaries = (
        df.groupby(["classifier", "dataset"])
        .describe()["elapsed_s"]
        .drop(columns="count")
        .sort_values(by="max", ascending=False)  # type: ignore
    )
    print(summaries.applymap(to_readable))
