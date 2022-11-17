from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from umap import UMAP
from xgboost import XGBClassifier

from src.dataset import Dataset

if __name__ == "__main__":
    ds = list(filter(lambda ds: "devna" in ds.name, Dataset.load_all()))[0]
    # ds = list(filter(lambda ds: "ldpa" in ds.name, Dataset.load_all()))[0]
    print("Loading...", end="", flush=True)
    start = time()
    X = ds.data.drop(columns="__target").to_numpy()
    y = ds.data["__target"].to_numpy()
    y = LabelEncoder().fit_transform(y).astype(np.float64)
    # y /= y.max()
    elapsed = time() - start
    print(f"done. Took {int(elapsed)} seconds. ")
    X = StandardScaler().fit_transform(X)

    # start = time()
    # rf = RandomForestClassifier()
    # print("Fitting with RandomForest...", end="", flush=True)
    # rf.fit(X, y)
    # elapsed = time() - start
    # print(f"done. Took {int(elapsed)} seconds. ")

    xgb = XGBClassifier(enable_categorical=True, tree_method="hist", verbose=3, n_jobs=1)
    print("Fitting with XGBoost...", end="", flush=True)
    xgb.fit(
        X,
        y,
    )
    elapsed = time() - start
    print(f"done. Took {int(elapsed)} seconds. ")
    sys.exit()

    umap = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        verbose=True,
    )
    start = time()
    X_reduced = umap.fit_transform(X)
    duration = time() - start
    print(f"Took {duration} seconds for ndim=2")

    plt.scatter(*X_reduced.T, s=0.1, c=y, cmap="Spectral", alpha=1.0)
    plt.show()

    ndims = [int(p * X.shape[1]) for p in [0.25, 0.50, 0.75]]
    for ndim in ndims:
        umap = UMAP(
            n_neighbors=15,
            n_components=ndim,  # dimensionality
            min_dist=0.1,
            verbose=True,
        )
        start = time()
        X_reduced = umap.fit_transform(X)
        duration = time() - start
        print(f"Took {duration} seconds for ndim={ndim}")
