from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

from joblib import Parallel, delayed
from tqdm import tqdm
from typing_extensions import Literal


class TQDMParallel(Parallel):
    def __init__(
        self,
        n_jobs=None,
        backend=None,
        verbose=0,
        timeout=None,
        pre_dispatch="2 * n_jobs",
        batch_size="auto",
        temp_folder=None,
        max_nbytes="1M",
        mmap_mode="r",
        prefer=None,
        require=None,
        desc: Optional[str] = None,
    ):
        super().__init__(
            n_jobs,
            backend,
            verbose,
            timeout,
            pre_dispatch,
            batch_size,
            temp_folder,
            max_nbytes,
            mmap_mode,
            prefer,
            require,
        )
        self.desc = desc

    def print_progress(self):
        update = self.n_completed_tasks - self.pbar.n
        self.pbar.update(update)
        return super().print_progress()

    def __call__(self, iterable):
        if hasattr(iterable, "__len__"):
            self.pbar = tqdm(total=len(iterable), desc=self.desc)
        else:
            self.pbar = tqdm(desc=self.desc)
        return super().__call__(iterable)


def joblib_map(
    function: Callable,
    args: Iterable,
    max_workers: int = -1,
    desc: Optional[str] = None,
    verbose: int = 0,
    **kwargs: Any,
) -> Any:
    if "verbose" in kwargs:
        kwargs["verbose"] = verbose
    return TQDMParallel(n_jobs=max_workers, desc=desc, **kwargs)(
        [delayed(function)(arg) for arg in args]
    )
