from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.hparams.hparams import ContinuousHparam, Hparams, OrdinalHparam
from test.helpers import random_hparams

ROOT = Path(__file__).resolve().parent.parent  # isort: skip
DIR = ROOT / "__test_temp__"
DIR.mkdir(exist_ok=True)


def test_randoms() -> None:
    rng1 = np.random.default_rng(seed=69)
    rng2 = np.random.default_rng(seed=69)
    for _ in range(50):
        hps1 = Hparams(random_hparams(rng1))
        hps2 = Hparams(random_hparams(rng2))
        assert hps1 == hps2

        for i in range(50):
            qr1 = hps1.quasirandom(i, rng=rng1)
            qr2 = hps2.quasirandom(i, rng=rng2)
            assert qr1 == qr2


def test_discrepancies() -> None:
    c1 = ContinuousHparam("c1", value=None, max=10, min=0)
    c2 = ContinuousHparam("c2", value=None, max=10, min=0)
    o1 = OrdinalHparam("o1", value=None, max=10, min=0)
    o2 = OrdinalHparam("o2", value=None, max=10, min=0)
    hps = Hparams([c1, c2, o1, o2])
    for N in [50, 100, 200]:
        for _ in range(3):
            c1r, c2r, o1r, o2r = [], [], [], []
            c1q, c2q, o1q, o2q = [], [], [], []
            seed = np.random.randint(0, int(1e7))
            rng = np.random.default_rng(seed)
            rng2 = np.random.default_rng(seed)
            for i in range(N):
                hpr = hps.random(rng)
                c1r.append(hpr.hparams["c1"].value)
                c2r.append(hpr.hparams["c2"].value)
                o1r.append(hpr.hparams["o1"].value + np.random.uniform(0, 0.5))  # type: ignore # noqa
                o2r.append(hpr.hparams["o2"].value + np.random.uniform(0, 0.5))  # type: ignore # noqa

            for i in range(N):
                hpq = hps.quasirandom(iteration=i, rng=rng2)
                c1q.append(hpq.hparams["c1"].value)
                c2q.append(hpq.hparams["c2"].value)
                o1q.append(hpq.hparams["o1"].value + np.random.uniform(0, 0.5))  # type: ignore # noqa
                o2q.append(hpq.hparams["o2"].value + np.random.uniform(0, 0.5))  # type: ignore # noqa

            fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            axes[0][0].scatter(c1r, c2r, color="black", s=0.5)
            axes[0][0].set_title("Continous Random")

            axes[1][0].scatter(c1q, c2q, color="black", s=0.5)
            axes[1][0].set_title("Continous QuasiRandom")

            axes[0][1].scatter(o1r, o2r, color="black", s=0.5)
            axes[0][1].set_title("Ordinal Random")

            axes[1][1].scatter(o1q, o2q, color="black", s=0.5)
            axes[1][1].set_title("Ordinal QuasiRandom")
            fig.suptitle(f"{N} samples")
            fig.set_size_inches(10, 10)
            plt.show()
