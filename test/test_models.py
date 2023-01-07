from pathlib import Path

from src.dataset import Dataset
from src.enumerables import RuntimeClass
from src.models.logistic import LRModel

ROOT = Path(__file__).resolve().parent.parent  # isort: skip
DIR = ROOT / "__test_temp__"
DIR.mkdir(exist_ok=True)


def test_lr() -> None:
    for dsname in RuntimeClass.Fast.members()[:2]:
        ds = Dataset(dsname)
        LRModel()
        X_train, y_train, X_test, y_test = ds.get_monte_carlo_splits()
