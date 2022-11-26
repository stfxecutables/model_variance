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
from warnings import filterwarnings

from pandas.errors import PerformanceWarning
from tqdm import tqdm

from src.dataset import Dataset, reduce_categoricals
from src.enumerables import RuntimeClass


def reduce_categoricals_failsafe(ds: Dataset) -> None:
    try:
        reduce_categoricals(ds)
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e} for dataset {ds.name.name}")
        print(ds)
        return None


def compute_categorical_embeddings(runtime: RuntimeClass) -> None:
    datasets = [Dataset(name) for name in runtime.members()]
    desc = "Computing embeddings: {ds}"
    pbar = tqdm(datasets, desc=desc.format(ds=""))
    for ds in pbar:
        pbar.set_description(desc.format(ds=str(ds)))
        reduce_categoricals_failsafe(ds)
    pbar.close()


if __name__ == "__main__":
    filterwarnings("ignore", category=PerformanceWarning)
    compute_categorical_embeddings(RuntimeClass.Fast)
    compute_categorical_embeddings(RuntimeClass.Mid)
    compute_categorical_embeddings(RuntimeClass.Slow)
