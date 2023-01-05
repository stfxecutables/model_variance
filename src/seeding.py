from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
import os

from numpy.random import SeedSequence

from src.constants import SEEDS


def urandom_int() -> int:
    entropy = os.urandom(256)
    return int.from_bytes(entropy, byteorder="little", signed=False)


def generate_seed_sequences(n_seqs: int) -> list[SeedSequence]:
    seed = urandom_int()
    seqs = SeedSequence(entropy=seed).spawn(n_seqs)
    return seqs


def save_seed_seqs(seed_seqs: list[SeedSequence], outname: str) -> None:
    d = {}
    for i, ss in enumerate(seed_seqs):
        d[i] = dict(entropy=ss.entropy, spawn_key=list(ss.spawn_key))
    out = SEEDS / outname
    with open(out, "w") as fp:
        json.dump(d, fp)
    print(f"Saved seed sequences to {out}")


def load_seed_seqs(outfile: Path) -> dict[int, SeedSequence]:
    with open(outfile, "r") as fp:
        d: dict[int, dict[str, int | tuple[int, ...]]] = json.load(fp)
    seqs = {}
    for i, args in d.items():
        seqs[int(i)] = SeedSequence(
            entropy=int(args["entropy"]), spawn_key=tuple(args["spawn_key"])
        )
    return seqs


if __name__ == "__main__":
    seqs = generate_seed_sequences(50)
    print()
