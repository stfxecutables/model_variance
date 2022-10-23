from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
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
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

JSONS = ROOT / "data/json"

class Dataset:
    def __init__(self, path: Path) -> None:
        self.path = path
        stem = self.path.stem
        res = re.search(r"([0-9])_v([0-9]+)_(.*)", stem)
        if res is None:
            raise ValueError("Impossible", self.path)
        self.did = int(res[1])
        self.version = int(res[2])
        self.name = str(res[3]).lower()

    @staticmethod
    def load_all() -> List[Dataset]:
        pass


if __name__ == "__main__":
    jsons = sorted(JSONS.rglob("*.json"))
    for json in jsons:
        ds = Dataset(json)
        print(ds.did, ds.version, ds.name)