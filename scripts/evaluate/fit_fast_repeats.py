from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
from argparse import Namespace
from pathlib import Path
from time import time

from sklearn.model_selection import ParameterGrid

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
        rep=list(range(10)),
        run=list(range(10)),
        cont=CONT_PERTURBS,
        cat=[0, 0.1],
        level=["sample", "label"],
    )

    ARGS = [Namespace(**d) for d in list(ParameterGrid(GRID))]
    if os.environ.get("CC_CLUSTER") is None:
        print(f"Run will require {len(ARGS)} iterations.")
        sys.exit(1)
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
        pass
