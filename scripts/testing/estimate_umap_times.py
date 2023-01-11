from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from math import ceil
from pathlib import Path
from time import time
from traceback import print_exc
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import CategoricalDtype, DataFrame
from pandas.errors import PerformanceWarning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
from umap import UMAP
from xgboost import XGBClassifier

from src.constants import CAT_REDUCED, CONT_REDUCED
from src.dataset import Dataset
from src.enumerables import RuntimeClass

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
    cats = df.select_dtypes(include=[CategoricalDtype])  # type: ignore
    if cats.shape[1] == 0:
        return OneHotEncoder().fit_transform(cats).astype(np.float64)
    x = pd.get_dummies(cats).astype(np.float64).to_numpy()
    umap = UMAP(n_components=2, metric="jaccard")
    with catch_warnings():
        filterwarnings("ignore", message="gradient function", category=UserWarning)
        reduced = umap.fit_transform(x)  # type: ignore
    np.save(outfile, reduced)
    return reduced


def embed_continuous(ds_perc: tuple[Dataset, int]) -> NDArray[np.float64] | None:
    ds, percent = ds_perc
    outfile = CONT_REDUCED / f"{ds.name.name}_{percent}percent.npy"
    if outfile.exists():
        reduced: NDArray = np.load(outfile)
        return reduced

    df = ds.data.drop(columns="__target")
    X_float = df.select_dtypes(exclude=[CategoricalDtype]).astype(  # type: ignore
        np.float64
    )
    X_float -= X_float.mean(axis=0)
    X_float /= X_float.std(axis=0)

    n_components = ceil((percent / 100) * X_float.shape[1])
    umap = UMAP(n_components=n_components)  # type: ignore
    reduced = umap.fit_transform(X_float)  # type: ignore
    np.save(outfile, reduced)
    return reduced


def estimate_cat_embed_time(ds: Dataset) -> DataFrame | None:
    try:
        start = time()
        embed_categoricals(ds)
        embed_time = time() - start
        return DataFrame(
            {
                "dataset": ds.name.name,
                "n_samples": ds.n_samples,
                "n_cats": ds.n_cats,
                "embed_time": embed_time,
            },
            index=[0],
        )

    except Exception as e:
        print_exc()
        print(f"Got error: {e} for {ds.name}")
        return None


def estimate_continuous_embed_time(ds_perc: tuple[Dataset, int]) -> DataFrame | None:
    try:
        ds, percent = ds_perc
        start = time()
        embed_continuous(ds_perc)
        embed_time = time() - start
        return DataFrame(
            {
                "dataset": ds.name.name,
                "n_samples": ds.n_samples,
                "n_cats": ds.n_cats,
                "percent": percent,
                "embed_time": embed_time,
            },
            index=[0],
        )

    except Exception as e:
        print_exc()
        print(f"Got error: {e} for {ds.name}")  # type: ignore
        return None


def compute_estimate_categorical_embedding_times(runtime: RuntimeClass) -> None:
    outfile = ROOT / f"cat_embed_{runtime.value}_times.json"
    if outfile.exists():
        return
    datasets = [Dataset(name) for name in runtime.members()]
    runtimes = []
    desc = "Computing embeddings: {ds}"
    pbar = tqdm(datasets, desc=desc.format(ds=""))
    ds: Dataset
    for ds in pbar:
        # /gpfs/fs0/scratch/j/jlevman/dberger/model_variance/.venv/lib/python3.9/site-packages/umap/umap_.py:132:
        # UserWarning: A large number of your vertices were disconnected
        # from the manifold.
        # You might consider using find_disconnected_points() to find and
        # remove these points from your data.

        pbar.set_description(desc.format(ds=str(ds)))
        elapsed = estimate_cat_embed_time(ds)
        if elapsed is not None:
            runtimes.append(elapsed)
    pbar.close()
    runtimes = pd.concat(runtimes, ignore_index=True, axis=0)
    runtimes.to_json(outfile)
    print(f"Saved runtimes DataFrame to {outfile}")


def compute_estimate_continuous_embedding_times(
    runtime: RuntimeClass, percent: int
) -> None:
    outfile = ROOT / f"cont_embed_{percent}percent_{runtime.value}_times.json"
    if outfile.exists():
        return
    datasets = [Dataset(name) for name in runtime.members()]
    ds_percs = [(ds, percent) for ds in datasets]
    runtimes = []
    desc = "Computing continuous embeddings: {ds}"
    pbar = tqdm(ds_percs, desc=desc.format(ds=""))
    for ds_perc in pbar:
        ds, perc = ds_perc
        pbar.set_description(desc.format(ds=f"{ds}@{perc}%"))
        elapsed = estimate_continuous_embed_time(ds_perc)
        if elapsed is not None:
            runtimes.append(elapsed)
    pbar.close()
    runtimes = pd.concat(runtimes, ignore_index=True, axis=0)
    runtimes.to_json(outfile)
    print(f"Saved continuous embedding runtimes DataFrame to {outfile}")


def get_float_X(df: DataFrame) -> DataFrame:
    X = df.drop(columns="__target").convert_dtypes(
        infer_objects=False,
        convert_string=True,
        convert_integer=False,
        convert_floating=True,  # type: ignore
    )
    cat_dtypes = list(
        filter(
            lambda dtype: isinstance(dtype, CategoricalDtype),
            X.dtypes.unique().tolist(),
        )
    )
    if len(cat_dtypes) != 0:  # means some categoricals
        X_float = X.select_dtypes(exclude=cat_dtypes).astype(np.float64)
        X_float -= X_float.mean(axis=0)
        X_float /= X_float.std(axis=0)
        X_cat = X.select_dtypes(include=cat_dtypes)
        df = pd.concat([X_float, X_cat], axis=1).copy()
        return df

    # now must all be non-categorical
    df = X.astype(np.float64)
    return df


def estimate_runtime_xgb(dataset: Dataset) -> DataFrame | None:
    try:
        start = time()
        df = dataset.data
        loadtime = time() - start

        X = get_float_X(df)
        y = df["__target"]
        y = LabelEncoder().fit_transform(y).astype(np.float64)  # type: ignore

        n_jobs = 1 if len(X) < 50000 else 80
        xgb = XGBClassifier(enable_categorical=True, tree_method="hist", n_jobs=n_jobs)
        xgb.fit(X, y)
        elapsed = (time() - start) / 60
        return DataFrame(
            {"classifier": "xgb_hist", "load_secs": loadtime, "fit_minutes": elapsed},
            index=[0],
        )
    except Exception as e:
        print_exc()
        print(f"Got error: {e} on dataset: {dataset.name} (id={dataset.did})")
        print(dataset)
        return None


def check_conversions(dataset: Dataset) -> None:
    try:
        df = dataset.data
        get_float_X(df)
        y = df["__target"]
        LabelEncoder().fit_transform(y).astype(np.float64)  # type: ignore
    except Exception as e:
        print_exc()
        print(f"Got error: {e} on dataset: {dataset.name} (id={dataset.did})")
        print(dataset)


if __name__ == "__main__":
    compute_estimate_categorical_embedding_times(RuntimeClass.Fast)
    compute_estimate_categorical_embedding_times(RuntimeClass.Mid)
    compute_estimate_categorical_embedding_times(RuntimeClass.Slow)

    # problem datasets:
    # Kr_vs_kp (after dropping const, all categorical, useless )
    # Car (all categorical, needs dropping)
    compute_estimate_continuous_embedding_times(RuntimeClass.Fast, percent=25)
    compute_estimate_continuous_embedding_times(RuntimeClass.Fast, percent=50)
    compute_estimate_continuous_embedding_times(RuntimeClass.Fast, percent=75)

    # problem datasets:
    # Vehice ???
    compute_estimate_continuous_embedding_times(RuntimeClass.Mid, percent=25)
    compute_estimate_continuous_embedding_times(RuntimeClass.Mid, percent=50)
    compute_estimate_continuous_embedding_times(RuntimeClass.Mid, percent=75)

    # problem datasets:
    compute_estimate_continuous_embedding_times(RuntimeClass.Slow, percent=25)
    compute_estimate_continuous_embedding_times(RuntimeClass.Slow, percent=50)
    compute_estimate_continuous_embedding_times(RuntimeClass.Slow, percent=75)
