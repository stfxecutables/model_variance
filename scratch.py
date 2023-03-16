from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from numpy import ndarray
from tqdm import tqdm

from src.constants import RESULTS
from src.dataset import Dataset
from src.enumerables import DatasetName, HparamPerturbation, PerturbMagnitude
from src.hparams.hparams import OrdinalHparam


def sig_perturb(x: ndarray, n_digits: int = 1) -> ndarray:
    delta = 10 ** (np.floor(np.log10(np.abs(x))) / 10**n_digits)
    if n_digits == 1:
        delta *= 2

    return x + delta * np.random.uniform(-1, 1, x.shape)


def perc_perturb_norm(lr: float) -> float:
    mn, mx = np.log10(1e-7), np.log10(1)
    d = mx - mn
    lrn = (np.log10(lr) - mn) / d
    delta = 0.05
    lr_min = 10 ** ((lrn - delta) * d + mn)
    lr_max = 10 ** ((lrn + delta) * d + mn)
    print(f"lr={lr:0.2e}: [{lr_min:0.2e}, {lr_max:0.2e}]")


def perc_perturb(lr: float) -> float:
    mn, mx = np.log10(1e-7), np.log10(1)
    lrn = np.log10(lr)
    d = mx - mn
    delta = d * 0.05
    lr_min = 10 ** ((lrn - delta))
    lr_max = 10 ** ((lrn + delta))
    print(f"lr={lr:0.2e}: [{lr_min:0.2e}, {lr_max:0.2e}]")


def sig_perturb_demo() -> None:
    x = np.full([5000, 5000], np.random.uniform(0, 1000))
    print(f"Original x value: {x[0][0]:0.4e}")
    for N_DIGITS in [0, 1, 2, 3]:
        pert = sig_perturb(x, n_digits=N_DIGITS)
        pmax = pert.max()
        pmin = pert.min()
        ps = np.array([pmin, pmax])
        np.set_printoptions(
            formatter={
                "float": lambda x: np.format_float_scientific(
                    x, precision=N_DIGITS, trim="k"
                )
            }
        )
        nth = {
            0: "zeroth",
            1: "first",
            2: "second",
            3: "third",
        }[N_DIGITS]
        print(f"Showing / perturbing at {nth} significant digit")
        print("              x:", x[0][0].reshape(-1, 1))
        # N2 = N_DIGITS if N_DIGITS > 1 else N_DIGITS + 1
        N2 = N_DIGITS + 1
        np.set_printoptions(
            formatter={
                "float": lambda x: np.format_float_scientific(x, precision=N2, trim="k")
            }
        )
        print("perturbed range: ", ps)
        print("\n")
    sys.exit()


def plot_skewed_data() -> None:
    for dsname in tqdm([*DatasetName]):
        try:
            ds = Dataset(dsname)
            outdir = RESULTS / "plots/feature_dists"
            outdir.mkdir(exist_ok=True, parents=True)
            outfile = outdir / f"{ds.name.name}.png"
            X = ds.get_X_continuous(reduction=None)
            y = ds.data["__target"]
            fig, ax = plt.subplots(nrows=1, ncols=1)
            for i in range(X.shape[1]):
                x = X[:, i]
                sbn.kdeplot(x, ax=ax, lw=0.5)
                # ax.hist(x, bins=100)
            xmin, xmax = np.percentile(X.ravel(), [2.5, 97.5])
            ax.vlines([xmin, xmax], ymin=0, ymax=ax.get_ylim()[1])
            fig.suptitle(ds.name.name)
            fig.set_size_inches(w=15, h=10)
            fig.savefig(outfile, dpi=300)
            print(f"Saved figure to {outfile}")
            plt.close()
        except:
            print("Did not plot: ", ds.name.name)
            continue


if __name__ == "__main__":

    o = OrdinalHparam("o", value=200, max=500, min=0)
    print(o)
    print(
        "AbsPercent",
        o.perturbed(
            method=HparamPerturbation.AbsPercent, magnitude=PerturbMagnitude.AbsPercent10
        ),
    )
    print(
        "RelPercent",
        o.perturbed(
            method=HparamPerturbation.RelPercent, magnitude=PerturbMagnitude.RelPercent10
        ),
    )
    print(
        "SigDig",
        o.perturbed(method=HparamPerturbation.SigDig, magnitude=PerturbMagnitude.SigOne),
    )
    # sig_perturb_demo()
    # print("Norm")
    # for lr in [1e-4, 2e-4, 5e-4]:
    #     perc_perturb_norm(lr)
    # print("No-Norm")
    # for lr in [1e-4, 2e-4, 5e-4]:
    #     perc_perturb(lr)

    # for dsname in [
    #     DatasetName.ClickPrediction,
    #     DatasetName.CreditCardFraud,
    #     DatasetName.Dionis,
    #     DatasetName.Fabert,
    #     DatasetName.Miniboone,
    #     DatasetName.Shuttle,
    # ]:
    #     ds = Dataset(dsname)
    #     X = ds.get_X_continuous()
    #     print(f"\n{ds.name.name}")
    #     print(
    #         DataFrame(X)
    #         .describe(percentiles=[0.01, 0.025, 0.05, 0.50, 0.95, 0.975, 0.99])
    #         .T.sort_values(by="max", ascending=False)
    #     )
    #     print(f"Percent values >  5sd: {np.mean(X.ravel() > 5)}")
    #     print(f"Percent values > 10sd: {np.mean(X.ravel() > 10)}")

    # mega outliers:
    #   ClickPrediction
    #   CreditCardFraud
    #   Dionis
    #   Fabert
    #   Miniboone
    #   Shuttle

    # ds = Dataset(DatasetName.Arrhythmia)
    # dists = ds.nearest_distances(reduction=None)

    # df = ds.data
    # X = df.drop(columns="__target")
    # # below is about 5 minutes with 8 cores, 2.5 minutes on 40 cores
    # nn = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    # start = time()
    # nn.fit(X)
    # dists, neighb_idx = nn.kneighbors(X, n_neighbors=2, return_distance=True)
    # # zeros are just the points themselves
    # dists = dists[:, 1]
    # neighb_idx = neighb_idx[:, 1]
    # elapsed = time() - start
    # print(f"Elapsed second: {elapsed}")
    # print()
