import numpy as np
from pytest import CaptureFixture
from tqdm import tqdm

from src.dataset import Dataset
from src.enumerables import DataPerturbation, DatasetName

N_REP = 10
NCOLS = 80
DESC = "{:<20}"


def test_categorical(capsys: CaptureFixture) -> None:
    for i, name in enumerate(DatasetName):
        if i == 0:
            continue
        ds = Dataset(name)
        if ds.n_categoricals <= 1:
            continue
        print(ds.name.name)
        for _ in range(10):
            X = ds.get_X_categorical(perturbation_prob=0, reduction=None)
            X_cat = ds.get_X_categorical(
                perturbation_prob=0.1, perturb_level="label", reduction=None
            )
            X_cat2 = ds.get_X_categorical(
                perturbation_prob=0.1, perturb_level="sample", reduction=None
            )
            print(np.mean(X != X_cat))
            print(np.mean(X != X_cat2))


def test_neighbour(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        pbar = tqdm(
            enumerate(DatasetName),
            DESC.format("Dataset"),
            total=len(DatasetName),
            ncols=NCOLS,
        )
        for i, name in pbar:
            ds = Dataset(name)
            pbar.set_description(DESC.format(ds.name.name))
            if ds.n_continuous <= 0:
                continue
            for _ in range(2):
                ds.get_X_continuous(
                    perturbation=DataPerturbation.HalfNeighbor,
                    reduction=None,
                )
                ds.get_X_continuous(
                    perturbation=DataPerturbation.QuarterNeighbor,
                    reduction=None,
                )


def test_sigdig(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        pbar = tqdm(
            enumerate(DatasetName),
            DESC.format("Dataset"),
            total=len(DatasetName),
            ncols=NCOLS,
        )
        for i, name in pbar:
            ds = Dataset(name)
            pbar.set_description(DESC.format(ds.name.name))
            if ds.n_continuous <= 0:
                continue
            for _ in range(N_REP):
                ds.get_X_continuous(
                    perturbation=DataPerturbation.SigDigOne,
                    reduction=None,
                )
                ds.get_X_continuous(
                    perturbation=DataPerturbation.SigDigZero,
                    reduction=None,
                )


def test_percentile(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        pbar = tqdm(
            enumerate(DatasetName),
            DESC.format("Dataset"),
            total=len(DatasetName),
            ncols=NCOLS,
        )
        for i, name in pbar:
            ds = Dataset(name)
            pbar.set_description(DESC.format(ds.name.name))
            if ds.n_continuous <= 0:
                continue
            for _ in range(N_REP):
                ds.get_X_continuous(
                    perturbation=DataPerturbation.Percentile05,
                    reduction=None,
                )
                ds.get_X_continuous(
                    perturbation=DataPerturbation.Percentile10,
                    reduction=None,
                )


def test_relative(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        pbar = tqdm(
            enumerate(DatasetName),
            DESC.format("Dataset"),
            total=len(DatasetName),
            ncols=NCOLS,
        )
        for i, name in pbar:
            ds = Dataset(name)
            pbar.set_description(DESC.format(ds.name.name))
            if ds.n_continuous <= 0:
                continue
            for _ in range(N_REP):
                ds.get_X_continuous(
                    perturbation=DataPerturbation.RelPercent05,
                    reduction=None,
                )
                ds.get_X_continuous(
                    perturbation=DataPerturbation.RelPercent10,
                    reduction=None,
                )