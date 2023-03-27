from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Sequence, Union

from src.enumerables import ClassifierKind
from src.hparams.hparams import (
    ContinuousHparam,
    FixedHparam,
    Hparam,
    Hparams,
    OrdinalHparam,
)


# https://neptune.ai/blog/lightgbm-parameters-guide
# https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
def lightgbm_hparams(
    n_estimators: Optional[int] = 200,
    reg_alpha: Optional[float] = 1e-8,
    reg_lambda: Optional[float] = 1e-8,
    num_leaves: Optional[int] = 31,
    colsample_bytree: Optional[float] = 0.95,
    subsample: Optional[int] = 0.95,
    subsample_freq: Optional[int] = 0,
    min_child_samples: Optional[int] = 20,
    # min_data_in_leaf: Optional[int] = None,
    # min_gain_to_split: Optional[] = None,
) -> List[Hparam]:
    return [
        OrdinalHparam(
            "n_estimators",
            n_estimators,
            max=250,
            min=50,
            default=200,
        ),
        ContinuousHparam(
            "reg_alpha",
            reg_alpha,
            max=10.0,
            min=1e-8,
            log_scale=True,
            default=1e-8,
        ),
        ContinuousHparam(
            "reg_lambda",
            reg_lambda,
            max=10.0,
            min=1e-8,
            log_scale=True,
            default=1e-8,
        ),
        OrdinalHparam(
            "num_leaves",
            num_leaves,
            max=256,
            min=2,
            default=31,
        ),
        ContinuousHparam(
            "colsample_bytree",
            colsample_bytree,
            max=1.0,
            min=0.4,
            log_scale=False,
            default=0.95,
        ),
        ContinuousHparam(
            "subsample",
            subsample,
            max=1.0,
            min=0.4,
            log_scale=False,
            default=0.95,
        ),
        OrdinalHparam(
            "subsample_freq",
            subsample_freq,
            max=7,
            min=0,
            default=0,
        ),
        OrdinalHparam(
            "min_child_samples",
            min_child_samples,
            max=100,
            min=5,
            default=20,
        ),
        FixedHparam(
            "objective",
            value="multiclass",
        ),
        FixedHparam("n_jobs", value=1),
    ]


class LightGBMHparams(Hparams):
    kind = ClassifierKind.LightGBM

    def __init__(
        self,
        hparams: Optional[Union[Collection[Hparam], Sequence[Hparam]]] = None,
    ) -> None:

        if hparams is None:
            hparams = lightgbm_hparams()
        super().__init__(hparams)


if __name__ == "__main__":
    gbm = LightGBMHparams()
    print(gbm)
