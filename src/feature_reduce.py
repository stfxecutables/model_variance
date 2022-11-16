from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from time import time

from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.dataset import Dataset

if __name__ == "__main__":
    ds = list(filter(lambda ds: "devna" in ds.name, Dataset.load_all()))[0]
    print("Loading...", end="", flush=True)
    start = time()
    X = ds.data.drop(columns="__target").to_numpy()
    elapsed = time() - start
    print(f"done. Took {int(elapsed)} seconds. ")
    X = StandardScaler().fit_transform(X)

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
