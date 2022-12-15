from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import traceback
from pathlib import Path
from time import sleep
from warnings import filterwarnings

from pandas.errors import PerformanceWarning
from tqdm import tqdm

from src.dataset import Dataset
from src.enumerables import RuntimeClass


def compute_distances_failsafe(ds_reduction: tuple[Dataset, int | None]) -> None:
    try:
        ds, reduction = ds_reduction
        ds.nearest_distances(reduction=reduction)
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e} for dataset {ds_reduction[0].name.name}")
        print(ds_reduction[0])
        sleep(2)
        return None


def compute_distances(runtime: RuntimeClass) -> None:
    datasets = [Dataset(name) for name in runtime.members()]
    ds_reductions = []
    for reduction in [None, 25, 50, 75]:
        for ds in datasets:
            ds_reductions.append((ds, reduction))
    desc = "Computing distances: {ds}(reduction={red})"
    pbar = tqdm(ds_reductions, desc=desc.format(ds="", red="?"))
    for ds_reduction in pbar:
        ds, red = ds_reduction
        pbar.set_description(desc.format(ds=str(ds), red=red))
        compute_distances_failsafe(ds_reduction)
    pbar.close()


if __name__ == "__main__":
    filterwarnings("ignore", category=PerformanceWarning)
    compute_distances(RuntimeClass.Fast)
    compute_distances(RuntimeClass.Mid)
    compute_distances(RuntimeClass.Slow)
