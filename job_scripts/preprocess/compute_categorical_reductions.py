from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import CategoricalDtype
from pandas.errors import PerformanceWarning
from tqdm import tqdm
from umap import UMAP

from src.constants import CAT_REDUCED
from src.dataset import Dataset
from src.enumerables import DatasetName, RuntimeClass

filterwarnings("ignore", category=PerformanceWarning)


def embed_categoricals(ds: Dataset) -> NDArray[np.float64] | None:
    """
    Notes
    -----
    We follow the guides:

        https://github.com/lmcinnes/umap/issues/58
        https://github.com/lmcinnes/umap/issues/104
        https://github.com/lmcinnes/umap/issues/241

    in spirit, but just embed all dummified categoricals to two dimensions.
    """
    outfile = CAT_REDUCED / f"{ds.name.name}.npy"
    if outfile.exists():
        reduced: NDArray = np.load(outfile)
        return reduced
    df = ds.data.drop(columns="__target")
    cats = df.select_dtypes(include=[CategoricalDtype])
    if cats.shape[1] == 0:
        return pd.get_dummies(cats).astype(np.float64).to_numpy()
    x = pd.get_dummies(cats).astype(np.float64).to_numpy()
    umap = UMAP(n_components=2, metric="jaccard")
    with catch_warnings():
        filterwarnings("ignore", message="gradient function", category=UserWarning)
        reduced = umap.fit_transform(x)
    np.save(outfile, reduced)
    return reduced


def compute_categorical_embeddings(runtime: RuntimeClass) -> None:
    datasets = [Dataset(name) for name in runtime.members()]
    desc = "Computing embeddings: {ds}"
    pbar = tqdm(datasets, desc=desc.format(ds=""))
    for ds in pbar:
        pbar.set_description(desc.format(ds=str(ds)))
        embed_categoricals(ds)
    pbar.close()


if __name__ == "__main__":
    # compute_categorical_embeddings(RuntimeClass.Fast)
    # compute_categorical_embeddings(RuntimeClass.Mid)
    compute_categorical_embeddings(RuntimeClass.Slow)
