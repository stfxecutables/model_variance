from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import traceback
from math import ceil
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import CategoricalDtype
from pandas.errors import PerformanceWarning
from tqdm import tqdm
from umap import UMAP

from src.constants import CONT_REDUCED
from src.dataset import Dataset
from src.enumerables import DatasetName, RuntimeClass

filterwarnings("ignore", category=PerformanceWarning)


def embed_continuous(ds_perc: tuple[Dataset, int]) -> NDArray[np.float64] | None:
    try:
        ds, percent = ds_perc
        # do not bother with all-categorical data here
        if ds.name in [DatasetName.Kr_vs_kp, DatasetName.Car, DatasetName.Connect4]:
            return None
        outfile = CONT_REDUCED / f"{ds.name.name}_{percent}percent.npy"
        if outfile.exists():
            reduced: NDArray = np.load(outfile)
            return reduced

        df = ds.data.drop(columns="__target")
        X_float = df.select_dtypes(exclude=[CategoricalDtype]).astype(np.float64)
        X_float -= X_float.mean(axis=0)
        X_float /= X_float.std(axis=0)

        n_components = ceil((percent / 100) * X_float.shape[1])
        umap = UMAP(n_components=n_components)
        reduced = umap.fit_transform(X_float)
        np.save(outfile, reduced)
        return reduced
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e} for dataset: {ds_perc[0].name.name}")
        print(ds_perc[0])


def compute_continuous_embeddings(runtime: RuntimeClass, percent: int) -> None:
    datasets = [Dataset(name) for name in runtime.members()]
    ds_percs = [(ds, percent) for ds in datasets]
    desc = "Computing continuous embeddings: {ds}"
    pbar = tqdm(ds_percs, desc=desc.format(ds=""))
    for ds_perc in pbar:
        ds, perc = ds_perc
        pbar.set_description(desc.format(ds=f"{ds}@{perc}%"))
        embed_continuous(ds_perc)
    pbar.close()


if __name__ == "__main__":
    # problem datasets:
    # Kr_vs_kp (after dropping const, all categorical, useless )
    # Car (all categorical, needs dropping)
    compute_continuous_embeddings(RuntimeClass.Fast, percent=25)
    compute_continuous_embeddings(RuntimeClass.Fast, percent=50)
    compute_continuous_embeddings(RuntimeClass.Fast, percent=75)

    # problem datasets:
    compute_continuous_embeddings(RuntimeClass.Mid, percent=25)
    compute_continuous_embeddings(RuntimeClass.Mid, percent=50)
    compute_continuous_embeddings(RuntimeClass.Mid, percent=75)

    # problem datasets:
    # SkinSegmentation: But still produces file?
    #     UserWarning: WARNING: spectral initialisation failed! The eigenvector
    #     solver failed. This is likely due to too small an eigengap. Consider
    #     adding some noise or jitter to your data.  Falling back to random
    #     initialisation!
    # Connect4: All categorical
    #     ValueError: at least one array or dtype is required
    compute_continuous_embeddings(RuntimeClass.Slow, percent=25)
    compute_continuous_embeddings(RuntimeClass.Slow, percent=50)
    compute_continuous_embeddings(RuntimeClass.Slow, percent=75)