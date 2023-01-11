from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
import os
from typing import Any

import numpy as np
from numpy.random import Generator, SeedSequence

from src.constants import DEFAULT_SEEDS, SEEDS


def urandom_int() -> int:
    entropy = os.urandom(256)
    return int.from_bytes(entropy, byteorder="little", signed=False)


def generate_seed_sequences(n_seqs: int) -> list[SeedSequence]:
    seed = urandom_int()
    seqs = SeedSequence(entropy=seed).spawn(n_seqs)
    return seqs


def save_seed_seqs(
    seed_seqs: list[SeedSequence], outname: str = DEFAULT_SEEDS.name
) -> None:
    d = {}
    for i, ss in enumerate(seed_seqs):
        d[i] = dict(entropy=ss.entropy, spawn_key=list(ss.spawn_key))
    out = SEEDS / outname
    with open(out, "w") as fp:
        json.dump(d, fp)
    print(f"Saved seed sequences to {out}")


def load_seed_seqs(outfile: Path = DEFAULT_SEEDS) -> dict[int, SeedSequence]:
    with open(outfile, "r") as fp:
        d: dict[int, Any] = json.load(fp)
    seqs = {}
    for i, args in d.items():
        seqs[int(i)] = SeedSequence(
            entropy=int(args["entropy"]),
            spawn_key=tuple(args["spawn_key"]),
        )
    return seqs


def load_rngs(seedfile: Path = DEFAULT_SEEDS) -> list[Generator]:
    seqs = load_seed_seqs(outfile=seedfile)
    return [np.random.default_rng(ss) for ss in seqs]


def load_repeat_rng(repeat: int, seedfile: Path = DEFAULT_SEEDS) -> Generator:
    seqs = load_seed_seqs(outfile=seedfile)
    ss = seqs[repeat]
    return np.random.default_rng(ss)


def load_run_rng(repeat: int, run: int, seedfile: Path = DEFAULT_SEEDS) -> Generator:
    seqs = load_seed_seqs(outfile=seedfile)
    parent = seqs[repeat]
    # below is safe to do because each element of spawn has same entropy, but with
    # spawn_key being an incremented tuple. E.g.
    #
    #   SeedSequence(e).spawn(n)[i] == SeedSequence(e).spawn(n + k)[i]
    #
    # for all i < n, and any k.
    ss = parent.spawn(run + 1)[-1]
    return np.random.default_rng(ss)


if __name__ == "__main__":
    if DEFAULT_SEEDS.exists():
        raise FileExistsError(
            f"Seeds already generated at {DEFAULT_SEEDS}. Remove it to re-generate seeds"
        )
    seqs = generate_seed_sequences(5000)
    save_seed_seqs(seqs)
