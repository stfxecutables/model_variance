from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


# Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ensure_dir(ROOT / "results")
CC_RESULTS = ensure_dir(ROOT / "cc_results")
TESTING_TEMP = ensure_dir(ROOT / "testing_temp")
DATA = ensure_dir(ROOT / "data")
LOGS = ensure_dir(ROOT / "logs")
CKPTS = ensure_dir(ROOT / "ckpts")
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
BATCH_SIZE = 1024

# hparam defaults
LR_LR_INIT_DEFAULT = 1e-2
LR_WD_DEFAULT = 0

MLP_LR_INIT_DEFAULT = 3e-4
MLP_WD_DEFAULT = 1e-4
DROPOUT_DEFAULT = 0.1
MLP_WIDTH_DEFAULT = 512

SKLEARN_SGD_LR_MIN = 1e-6
SKLEARN_SGD_LR_MAX = 1.0
SKLEARN_SGD_LR_DEFAULT = 1e-3

"""
Default in sklearn is 'scale' = 1 / (n_feat * X.var()). Since we standardize,
X.var() == 1.0. n_feat ranges from 4 to 2000, so 2.5e-1 to 5e-4, but most are
in [10, 800], i.e. 1e-1, 1.25e-3. We choose as default a value in this range.
"""
SVM_GAMMA = 1e-2
