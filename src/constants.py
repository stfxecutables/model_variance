from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


# Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ensure_dir(ROOT / "results")
DATA = ensure_dir(ROOT / "data")
LOGS = ensure_dir(ROOT / "logs")
DEBUG_LOGS = ensure_dir(ROOT / "debug_logs")
PQS = ensure_dir(DATA / "parquet")
CAT_REDUCED = ensure_dir(PQS / "categorical_reductions")
CONT_REDUCED = ensure_dir(PQS / "continuous_reductions")
DISTANCES = ensure_dir(DATA / "precomputed")
SEEDS = ensure_dir(DATA / "seeds")
DEFAULT_SEEDS = SEEDS / "seed_seqs.json"

# constants
TEST_SIZE = 0.25
# LR_MAX_EPOCHS = 5
LR_MAX_EPOCHS = 10
MLP_MAX_EPOCHS = 20
"""Tests show this is quite good for the Runtime.Fast datasets"""
BATCH_SIZE = 64
