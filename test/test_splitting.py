from argparse import Namespace

import numpy as np
from pytest import CaptureFixture
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.dataset import Dataset
from src.enumerables import DatasetName, RuntimeClass


class TestReproducibility:
    def test_no_perturb(self, capsys: CaptureFixture) -> None:
        grid = dict(
            dsname=RuntimeClass.Fast.members()[:2],
            down=[None, 25, 50, 75],
            red=[None, 25, 50, 75],
            rep=list(range(2)),
            run=list(range(2)),
        )

        args = [Namespace(**d) for d in list(ParameterGrid(grid))]
        with capsys.disabled():
            fmt = "{dsname}: train_size={down}, reduce={red}, rep={rep}, run={run}"
            print("")
            pbar = tqdm(args, desc="", total=len(args), ncols=120)
            try:
                for arg in pbar:
                    pbar.set_description(
                        fmt.format(
                            dsname=arg.dsname,
                            down=arg.down,
                            red=arg.red,
                            rep=arg.rep,
                            run=arg.run,
                        )
                    )

                    ds = Dataset(arg.dsname)
                    X_train, y_train, X_test, y_test = ds.get_monte_carlo_splits(
                        train_downsample=arg.down,
                        cont_perturb=None,
                        cat_perturb_prob=0,
                        reduction=arg.red,
                        repeat=arg.rep,
                        run=arg.run,
                    )
                    X_train2, y_train2, X_test2, y_test2 = ds.get_monte_carlo_splits(
                        train_downsample=arg.down,
                        cont_perturb=None,
                        cat_perturb_prob=0,
                        reduction=arg.red,
                        repeat=arg.rep,
                        run=arg.run,
                    )
                    np.testing.assert_array_equal(X_train, X_train2)
                    np.testing.assert_array_equal(y_train, y_train2)
                    np.testing.assert_array_equal(X_test, X_test2)
                    np.testing.assert_array_equal(y_test, y_test2)
                    pbar.update()
            except Exception as e:
                raise e
            finally:
                pbar.clear()
                pbar.close()
