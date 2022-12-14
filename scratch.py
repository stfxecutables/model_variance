from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import time
from traceback import print_exc
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
from warnings import catch_warnings, filterwarnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from numpy.typing import NDArray
from pandas import CategoricalDtype, DataFrame, Series
from pandas.errors import PerformanceWarning
from scipy.spatial import Voronoi
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal
from umap import UMAP
from umap.utils import disconnected_vertices
from xgboost import XGBClassifier

from src.constants import CAT_REDUCED
from src.dataset import Dataset
from src.enumerables import DatasetName, RuntimeClass

if __name__ == "__main__":
    ds = Dataset(DatasetName.DevnagariScript)
    df = ds.data
    X = df.drop(columns="__target")
    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    start = time()
    nn.fit(X)
    neighbs = nn.kneighbors(X, n_neighbors=2, return_distance=True)
    elapsed = time() - start
    print(f"Elapsed second: {elapsed}")
    print()
