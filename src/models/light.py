from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union
from warnings import filterwarnings

import numpy as np
from lightgbm import LGBMClassifier
from lightgbm.sklearn import (
    LGBMModel,
    LGBMNotFittedError,
    _LGBMAssertAllFinite,
    _LGBMCheckClassificationTargets,
    _LGBMClassifierBase,
    _LGBMLabelEncoder,
    _lgbmmodel_doc_predict,
    _log_warning,
    log_evaluation,
)
from numpy import ndarray

from src.dataset import Dataset
from src.enumerables import ClassifierKind, DatasetName, RuntimeClass
from src.hparams.light import LightGBMHparams
from src.models.model import ClassifierModel


class FixedLGBMClassifier(_LGBMClassifierBase, LGBMModel):
    """Implementation is bugged and does not encode num_classes correctly"""

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, Callable]] = None,
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: float = 0,
        min_child_weight: float = 0.001,
        min_child_samples: int = 20,
        subsample: float = 1,
        subsample_freq: int = 0,
        colsample_bytree: float = 1,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: int = -1,
        silent: Union[bool, str] = "warn",
        importance_type: str = "split",
        **kwargs,
    ):
        self.all_class_labels = kwargs.pop("all_class_labels")
        super().__init__(
            boosting_type,
            num_leaves,
            max_depth,
            learning_rate,
            n_estimators,
            subsample_for_bin,
            objective,
            class_weight,
            min_split_gain,
            min_child_weight,
            min_child_samples,
            subsample,
            subsample_freq,
            colsample_bytree,
            reg_alpha,
            reg_lambda,
            random_state,
            n_jobs,
            silent,
            importance_type,
            **kwargs,
        )

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_class_weight=None,
        eval_init_score=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose="warn",
        feature_name="auto",
        categorical_feature="auto",
        # callbacks=[log_evaluation(period=-1)],
        callbacks=[],
        init_model=None,
    ):
        _LGBMAssertAllFinite(y)
        _LGBMCheckClassificationTargets(y)
        self._le = _LGBMLabelEncoder().fit(self.all_class_labels)
        try:
            _y = self._le.transform(y)
        except TypeError as e:
            raise RuntimeError(
                "Bullshit warning about ufunc 'isnan'. self.all_class_labels:\n"
                f"{self.all_class_labels}"
            ) from e
        self._class_map = dict(
            zip(self._le.classes_, self._le.transform(self._le.classes_))
        )
        if isinstance(self.class_weight, dict):
            self._class_weight = {
                self._class_map[k]: v for k, v in self.class_weight.items()
            }

        self._classes = self._le.classes_
        self._n_classes = len(self._classes)

        if self._n_classes > 2:
            # Switch to using a multiclass objective in the underlying LGBM instance
            ova_aliases = {"multiclassova", "multiclass_ova", "ova", "ovr"}
            if self._objective not in ova_aliases and not callable(self._objective):
                self._objective = "multiclass"

        if not callable(eval_metric):
            if isinstance(eval_metric, (str, type(None))):
                eval_metric = [eval_metric]
            if self._n_classes > 2:
                for index, metric in enumerate(eval_metric):
                    if metric in {"logloss", "binary_logloss"}:
                        eval_metric[index] = "multi_logloss"
                    elif metric in {"error", "binary_error"}:
                        eval_metric[index] = "multi_error"
            else:
                for index, metric in enumerate(eval_metric):
                    if metric in {"logloss", "multi_logloss"}:
                        eval_metric[index] = "binary_logloss"
                    elif metric in {"error", "multi_error"}:
                        eval_metric[index] = "binary_error"

        # do not modify args, as it causes errors in model selection tools
        valid_sets = None
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            valid_sets = [None] * len(eval_set)
            for i, (valid_x, valid_y) in enumerate(eval_set):
                if valid_x is X and valid_y is y:
                    valid_sets[i] = (valid_x, _y)
                else:
                    valid_sets[i] = (valid_x, self._le.transform(valid_y))
        super().fit(
            X,
            _y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=valid_sets,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_class_weight=eval_class_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )
        return self

    _base_doc = LGBMModel.fit.__doc__
    _base_doc = (
        _base_doc[: _base_doc.find("group :")]  # type: ignore
        + _base_doc[_base_doc.find("eval_set :") :]
    )  # type: ignore
    fit.__doc__ = (
        _base_doc[: _base_doc.find("eval_group :")]
        + _base_doc[_base_doc.find("eval_metric :") :]
    )

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        **kwargs,
    ):
        """Docstring is inherited from the LGBMModel."""
        result = self.predict_proba(
            X,
            raw_score,
            start_iteration,
            num_iteration,
            pred_leaf,
            pred_contrib,
            **kwargs,
        )
        if callable(self._objective) or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            # they also had this bugged and on the wrong axis in binary case

            class_index = np.argmax(result, axis=1)
            return self._le.inverse_transform(class_index)

    predict.__doc__ = LGBMModel.predict.__doc__

    def predict_proba(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        **kwargs,
    ):
        """Docstring is set after definition, using a template."""
        result = super().predict(
            X,
            raw_score,
            start_iteration,
            num_iteration,
            pred_leaf,
            pred_contrib,
            **kwargs,
        )
        if callable(self._objective) and not (raw_score or pred_leaf or pred_contrib):
            _log_warning(
                "Cannot compute class probabilities or labels "
                "due to the usage of customized objective function.\n"
                "Returning raw scores instead."
            )
            return result
        elif self._n_classes >= 2 or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            return np.vstack((1.0 - result, result)).transpose()

    predict_proba.__doc__ = _lgbmmodel_doc_predict.format(
        description="Return the predicted probability for each class for each sample.",
        X_shape="array-like or sparse matrix of shape = [n_samples, n_features]",
        output_name="predicted_probability",
        predicted_result_shape="array-like of shape = [n_samples] or shape = [n_samples, n_classes]",
        X_leaves_shape="array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]",
        X_SHAP_values_shape="array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects",
    )

    @property
    def classes_(self):
        """:obj:`array` of shape = [n_classes]: The class label array."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No classes found. Need to call fit beforehand.")
        return self._classes

    @property
    def n_classes_(self):
        """:obj:`int`: The number of classes."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No classes found. Need to call fit beforehand.")
        return self._n_classes


class LightGBMModel(ClassifierModel):
    def __init__(self, hparams: LightGBMHparams, logdir: Path, dataset: Dataset) -> None:
        super().__init__(hparams=hparams, logdir=logdir, dataset=dataset)
        self.kind: ClassifierKind = ClassifierKind.LightGBM
        self.hparams: LightGBMHparams
        self.model_cls: Type[FixedLGBMClassifier] = FixedLGBMClassifier
        self.model: FixedLGBMClassifier

    def predict(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        return np.array(self.model.predict(X)), y

    def _get_model_args(self) -> Dict[str, Any]:
        args = super()._get_model_args()
        runtime = RuntimeClass.from_dataset(self.dataset.name)
        if runtime in (RuntimeClass.Fast.members() + RuntimeClass.Mid.members()):
            args["n_jobs"] = 1
        elif runtime in RuntimeClass.Slow.members():
            args["n_jobs"] = 2
        args["num_classes"] = self.dataset.num_classes
        args["all_class_labels"] = self.dataset.get_encoded_y()
        return args


if __name__ == "__main__":
    for dsname in RuntimeClass.Fast.members()[:1]:
        data = Dataset(dsname)
        model = LightGBMModel(
            hparams=LightGBMHparams(), dataset=data, logdir=ROOT / "WTF"
        )
        X_train, y_train, X_test, y_test = data.get_monte_carlo_splits(
            train_downsample=None
        )
        model.fit(X_train, y_train, save=False)
        y_pred = model.predict(X_test, y_test)
        print(np.mean(y_pred == y_test))
