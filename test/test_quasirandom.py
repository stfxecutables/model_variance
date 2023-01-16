from math import ceil
from pathlib import Path

import numpy as np
from pytest import raises

from src.hparams.hparams import Hparams
from test.helpers import (
    random_categorical,
    random_continuous,
    random_fixed,
    random_hparams,
    random_ordinal,
)

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
