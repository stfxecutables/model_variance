from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

JSONS = ROOT / "data/json"


class DatasetName(Enum):
    Arrhythmia = "arrhythmia"
    Kc1 = "kc1"
    Click_prediction_small = "click_prediction_small"
    BankMarketing = "bank-marketing"
    BloodTransfusion = "blood-transfusion-service-center"
    Cnae9 = "cnae-9"
    Ldpa = "ldpa"
    Nomao = "nomao"
    Phoneme = "phoneme"
    SkinSegmentation = "skin-segmentation"
    WalkingActivity = "walking-activity"
    Adult = "adult"
    Higgs = "higgs"
    Numerai28_6 = "numerai28.6"
    Kr_vs_kp = "kr-vs-kp"
    Connect4 = "connect-4"
    Shuttle = "shuttle"
    DevnagariScript = "devnagari-script"
    Car = "car"
    Australian = "australian"
    Segment = "segment"
    FashionMnist = "fashion-mnist"
    JungleChess = "jungle_chess_2pcs_raw_endgame_complete"
    Christine = "christine"
    Jasmine = "jasmine"
    Sylvine = "sylvine"
    Miniboone = "miniboone"
    Dilbert = "dilbert"
    Fabert = "fabert"
    Volkert = "volkert"
    Dionis = "dionis"
    Jannis = "jannis"
    Helena = "helena"
    Aloi = "aloi"
    CreditCardFraud = "creditcardfrauddetection"
    Credit_g = "credit-g"
    Anneal = "anneal"
    MfeatFactors = "mfeat-factors"
    Vehicle = "vehicle"


class RuntimeClass(Enum):
    Fast = "fast"
    Mid = "medium"
    Slow = "slow"

    def members(self) -> list[DatasetName]:
        runtimes = {
            RuntimeClass.Fast: [  # <1min on one core
                DatasetName.Arrhythmia,
                DatasetName.Kc1,
                DatasetName.BankMarketing,
                DatasetName.BloodTransfusion,
                DatasetName.Cnae9,
                DatasetName.Nomao,
                DatasetName.Phoneme,
                DatasetName.Adult,
                DatasetName.Kr_vs_kp,
                DatasetName.Car,
                DatasetName.Australian,
                DatasetName.Segment,
                DatasetName.JungleChess,
                DatasetName.Christine,
                DatasetName.Jasmine,
                DatasetName.Sylvine,
                DatasetName.Dilbert,
                DatasetName.Fabert,
                DatasetName.Credit_g,
                DatasetName.Anneal,
                DatasetName.MfeatFactors,
                DatasetName.Vehicle,
            ],
            RuntimeClass.Mid: [  # 1-20 minutes on one core
                DatasetName.Dionis,
                DatasetName.Click_prediction_small,
                DatasetName.CreditCardFraud,
                DatasetName.SkinSegmentation,
                DatasetName.Ldpa,
                DatasetName.WalkingActivity,
                DatasetName.Miniboone,
                DatasetName.Aloi,
                DatasetName.Higgs,
                DatasetName.Numerai28_6,
            ],
            RuntimeClass.Slow: [  # 40-600+ minutes on one core
                DatasetName.DevnagariScript,  # <1 min with 80 Niagara cores
                DatasetName.Jannis,
                DatasetName.FashionMnist,
                DatasetName.Connect4,  # ~3 minutes with 80 Niagara cores
                DatasetName.Helena,
                DatasetName.Volkert,
                DatasetName.Shuttle,  # ~8 min with 80 Niagara cores
            ],
        }
        return runtimes[self]
