from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
import os
import sys
import traceback
from argparse import Namespace
from base64 import urlsafe_b64encode
from enum import Enum, EnumMeta
from shutil import rmtree
from time import strftime
from typing import Literal, Type, TypeVar
from uuid import UUID, uuid4

import numpy as np
from numpy import ndarray

from src.classifier import Classifier
from src.constants import DEBUG_LOGS, LOGS, ensure_dir
from src.dataset import Dataset
from src.enumerables import (
    ClassifierKind,
    DataPerturbation,
    DatasetName,
    HparamPerturbation,
    RuntimeClass,
)
from src.hparams.hparams import Hparam, Hparams
from src.models.logistic import LRModel
from src.models.mlp import MLPModel
from src.models.model import ClassifierModel
from src.models.svc import SVCModel
from src.models.xgb import XGBoostModel
from src.serialize import DirJSONable, FileJSONable
from src.utils import missing_keys
