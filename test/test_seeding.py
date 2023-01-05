import numpy as np

from src.constants import SEEDS
from src.seeding import generate_seed_sequences, load_seed_seqs, save_seed_seqs


def test_save() -> None:
    outfile = SEEDS / "test_save.json"
    try:
        ss = generate_seed_sequences(50)
        save_seed_seqs(ss, outname=outfile.name)
        assert outfile.exists()
    except Exception as e:
        raise e
    finally:
        outfile.unlink()


def test_load() -> None:
    outfile = SEEDS / "test_load.json"
    try:
        ss1 = generate_seed_sequences(50)
        save_seed_seqs(ss1, outname=outfile.name)
        ss2 = load_seed_seqs(outfile)
        rngs1 = [np.random.default_rng(ss) for ss in ss1]
        rngs2 = [np.random.default_rng(ss2[i]) for i in range(len(ss2))]
        for rng1, rng2 in zip(rngs1, rngs2):
            ints1 = rng1.integers(0, 100, 100)
            ints2 = rng2.integers(0, 100, 100)
            np.testing.assert_array_equal(ints1, ints2)

    except Exception as e:
        raise e
    finally:
        outfile.unlink()
