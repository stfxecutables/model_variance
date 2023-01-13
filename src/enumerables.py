from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from enum import Enum
from pathlib import Path

from sklearn.svm import SVC
from xgboost import XGBClassifier

JSONS = ROOT / "data/json"
PQS = ROOT / "data/parquet"


ThridPartyClassifierModel = SVC | XGBClassifier


class DatasetName(Enum):
    """Remove DevnagariScript and FashionMnist, which are not tabular data"""

    Arrhythmia = "arrhythmia"
    Kc1 = "kc1"
    ClickPrediction = "click_prediction_small"
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
    # DevnagariScript = "devnagari-script"
    Car = "car"
    Australian = "australian"
    Segment = "segment"
    # FashionMnist = "fashion-mnist"
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

    def path(self) -> Path:
        paths = {
            DatasetName.Arrhythmia: PQS / "1017_v2_arrhythmia.parquet",
            DatasetName.Kc1: PQS / "1067_v1_kc1.parquet",
            DatasetName.ClickPrediction: PQS / "1219_v4_Click_prediction_small.parquet",
            DatasetName.BankMarketing: PQS / "1461_v1_bank-marketing.parquet",
            DatasetName.BloodTransfusion: PQS
            / "1464_v1_blood-transfusion-service-center.parquet",
            DatasetName.Cnae9: PQS / "1468_v1_cnae-9.parquet",
            DatasetName.Ldpa: PQS / "1483_v1_ldpa.parquet",
            DatasetName.Nomao: PQS / "1486_v1_nomao.parquet",
            DatasetName.Phoneme: PQS / "1489_v1_phoneme.parquet",
            DatasetName.SkinSegmentation: PQS / "1502_v1_skin-segmentation.parquet",
            DatasetName.WalkingActivity: PQS / "1509_v1_walking-activity.parquet",
            DatasetName.Adult: PQS / "1590_v2_adult.parquet",
            DatasetName.Higgs: PQS / "23512_v2_higgs.parquet",
            DatasetName.Numerai28_6: PQS / "23517_v2_numerai28.6.parquet",
            DatasetName.Kr_vs_kp: PQS / "3_v1_kr-vs-kp.parquet",
            DatasetName.Connect4: PQS / "40668_v2_connect-4.parquet",
            DatasetName.Shuttle: PQS / "40685_v1_shuttle.parquet",
            # DatasetName.DevnagariScript: PQS / "40923_v1_Devnagari-Script.parquet",
            DatasetName.Car: PQS / "40975_v3_car.parquet",
            DatasetName.Australian: PQS / "40981_v4_Australian.parquet",
            DatasetName.Segment: PQS / "40984_v3_segment.parquet",
            # DatasetName.FashionMnist: PQS / "40996_v1_Fashion-MNIST.parquet",
            DatasetName.JungleChess: PQS
            / "41027_v1_jungle_chess_2pcs_raw_endgame_complete.parquet",
            DatasetName.Christine: PQS / "41142_v1_christine.parquet",
            DatasetName.Jasmine: PQS / "41143_v1_jasmine.parquet",
            DatasetName.Sylvine: PQS / "41146_v1_sylvine.parquet",
            DatasetName.Miniboone: PQS / "41150_v1_MiniBooNE.parquet",
            DatasetName.Dilbert: PQS / "41163_v1_dilbert.parquet",
            DatasetName.Fabert: PQS / "41164_v1_fabert.parquet",
            DatasetName.Volkert: PQS / "41166_v1_volkert.parquet",
            DatasetName.Dionis: PQS / "41167_v1_dionis.parquet",
            DatasetName.Jannis: PQS / "41168_v1_jannis.parquet",
            DatasetName.Helena: PQS / "41169_v1_helena.parquet",
            DatasetName.Aloi: PQS / "42396_v3_aloi.parquet",
            DatasetName.CreditCardFraud: PQS
            / "42397_v2_CreditCardFraudDetection.parquet",
            DatasetName.Credit_g: PQS / "44096_v2_credit-g.parquet",
            DatasetName.Anneal: PQS / "44268_v17_anneal.parquet",
            DatasetName.MfeatFactors: PQS / "978_v2_mfeat-factors.parquet",
            DatasetName.Vehicle: PQS / "994_v2_vehicle.parquet",
        }
        return paths[self]


class RuntimeClass(Enum):
    Fast = "fast"
    Mid = "medium"
    Slow = "slow"

    def members(self) -> list[DatasetName]:
        runtimes = {
            RuntimeClass.Fast: [  # <1min on one core
                DatasetName.Arrhythmia,
                DatasetName.Kc1,
                DatasetName.BloodTransfusion,
                DatasetName.Cnae9,
                DatasetName.Phoneme,
                DatasetName.Kr_vs_kp,
                DatasetName.Car,
                DatasetName.Australian,
                DatasetName.Segment,
                DatasetName.Christine,
                DatasetName.Jasmine,
                DatasetName.Sylvine,
                DatasetName.Credit_g,
                DatasetName.Anneal,
                DatasetName.MfeatFactors,
                DatasetName.Vehicle,
            ],
            # two of the below are very slow even with linear SVC
            RuntimeClass.Mid: [  # 1-20 minutes on one core
                DatasetName.JungleChess,  # <2min SVC
                DatasetName.Adult,  # <2min SVC
                DatasetName.Fabert,  # <2min SVC
                DatasetName.Dilbert,  # <2min SVC
                DatasetName.Nomao,  # 3-5min radial SVC
                DatasetName.BankMarketing,  # <1min radial SVC
                DatasetName.ClickPrediction,
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
                # DatasetName.DevnagariScript,  # <1 min with 80 Niagara cores
                DatasetName.Dionis, # PROBABLY THE PROBLEM
                DatasetName.Jannis,
                # DatasetName.FashionMnist,
                DatasetName.Connect4,  # ~3 minutes with 80 Niagara cores
                DatasetName.Helena,
                DatasetName.Volkert,
                DatasetName.Shuttle,  # ~8 min with 80 Niagara cores
            ],
        }
        return runtimes[self]

    @staticmethod
    def very_fasts() -> list[DatasetName]:
        return [
            DatasetName.Arrhythmia,
            DatasetName.Kc1,
            DatasetName.BloodTransfusion,
            DatasetName.Cnae9,
            DatasetName.Phoneme,
            DatasetName.Kr_vs_kp,
            DatasetName.Car,
            DatasetName.Australian,
            DatasetName.Segment,
            DatasetName.Jasmine,
            DatasetName.Sylvine,
            # DatasetName.Fabert,  # takes about 30s-1min on 2015 Macbook Pro for SVM
            DatasetName.Credit_g,
            DatasetName.Anneal,
            DatasetName.MfeatFactors,
            DatasetName.Vehicle,
        ]

    @staticmethod
    def from_dataset(dsname: DatasetName) -> RuntimeClass:
        cls: RuntimeClass
        for cls in RuntimeClass:
            if dsname in cls.members():
                return cls
        raise RuntimeError(f"Impossible! Invalid DatasetName: {dsname}")


# class HparamPerturbation(Enum):
#     SigDig = "sig-dig"
#     RelPercent = "rel-percent"
#     AbsPercent = "abs-percent"


class DataPerturbation(Enum):
    HalfNeighbor = "half-neighbour"
    QuarterNeighbor = "quarter-neighbour"
    SigDigZero = "sig0"
    SigDigOne = "sig1"
    RelPercent10 = "rel-10"
    RelPercent05 = "rel-05"
    Percentile10 = "percentile-10"
    Percentile05 = "percentile-05"


class HparamPerturbation(Enum):
    SigZero = "sig-zero"
    SigOne = "sig-one"
    Percentile10 = "percentile-10"
    Percentile05 = "percentile-05"
    RelPercent10 = "rel-percent-10"
    RelPercent05 = "rel-percent-05"
    AbsPercent10 = "abs-percent-10"
    AbsPercent05 = "abs-percent-05"

    def magnitude(self) -> int | float:
        return {
            HparamPerturbation.SigZero: 0,
            HparamPerturbation.SigOne: 1,
            HparamPerturbation.Percentile05: 5,
            HparamPerturbation.Percentile10: 10,
            HparamPerturbation.RelPercent05: 0.05,
            HparamPerturbation.RelPercent10: 0.10,
            HparamPerturbation.AbsPercent05: 0.05,
            HparamPerturbation.AbsPercent10: 0.10,
        }[self]


class ClassifierKind(Enum):
    XGBoost = "xgb"
    SGD_SVM = "svm-sgd"
    LinearSVM = "svm-linear"
    SVM = "svm"
    MLP = "mlp"
    LR = "lr"  # MLP with one linear layer, mathematically identical

    # def model(self) -> ClassifierModel:
    #     """Should return something that implements `.fit()` and `.predict()` methods"""
    #     models: dict[ClassifierKind, ClassifierModel] = {
    #         ClassifierKind.XGBoost: XGBClassifier,
    #         ClassifierKind.SVM: SVC,
    #         ClassifierKind.MLP: MLP,
    #         ClassifierKind.LR: LogisticRegression,
    #     }
    #     return models[self]
