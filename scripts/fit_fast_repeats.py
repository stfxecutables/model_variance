from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


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
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import CategoricalDtype, DataFrame, Series
from pandas.errors import PerformanceWarning
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal
from xgboost import XGBClassifier

from src.dataset import Dataset
from src.enumerables import DataPerturbation, RuntimeClass

if __name__ == "__main__":
    CONT_PERTURBS = [
        None,
        DataPerturbation.HalfNeighbor,
        DataPerturbation.SigDigOne,
        DataPerturbation.SigDigZero,
        DataPerturbation.RelPercent10,
    ]
    GRID = dict(
        dsname=RuntimeClass.Fast.members(),
        down=[None, 25, 50, 75],
        red=[None],  # we don't perturb UMAP reductions
        rep=list(range(2)),
        run=list(range(2)),
        cont=CONT_PERTURBS,
        cat=[0, 0.1],
        level=["sample", "label"],
    )

    ARGS = [Namespace(**d) for d in list(ParameterGrid(grid))]
    fmt = (
        "{dsname}: train_size={down}, reduce={red}, rep={rep}, "
        "run={run}, cont_pert={cont}, cat_pert={cat}, cat_level={level}"
    )
    print("")
    start = time()
    try:
        for i, arg in enumerate(ARGS):
            elapsed = (time() - start) / 60
            unit = "mins"
            if elapsed > 120:
                elapsed /= 60
                unit = "hrs"
            duration = f"{elapsed} {unit}"
            print(
                f"Evaluating iteration {i} of {len(ARGS)}. "
                f"Total elapsed time: {duration}"
            )
            print(
                fmt.format(
                    dsname=arg.dsname,
                    down=arg.down,
                    red=arg.red,
                    rep=arg.rep,
                    run=arg.run,
                    cont=arg.cont.name,
                    cat=arg.cat,
                    level=arg.level,
                )
            )

            ds = Dataset(arg.dsname)
            X_train, y_train, X_test, y_test = ds.get_monte_carlo_splits(
                train_downsample=arg.down,
                cont_perturb=arg.cont,
                cat_perturb_prob=arg.cat,
                cat_perturb_level=arg.level,
                reduction=arg.red,
                repeat=arg.rep,
                run=arg.run,
            )
    except Exception as e:
        raise e
    finally:
        pbar.clear()
        pbar.close()
