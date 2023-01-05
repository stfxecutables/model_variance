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

from typing_extensions import Literal


def ensure_dir(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ensure_dir(ROOT / "results")
DATA = ensure_dir(ROOT / "data")
PQS = ensure_dir(DATA / "parquet")
CAT_REDUCED = ensure_dir(PQS / "categorical_reductions")
CONT_REDUCED = ensure_dir(PQS / "continuous_reductions")
DISTANCES = ensure_dir(DATA / "precomputed")
SEEDS = ensure_dir(DATA / "seeds")
