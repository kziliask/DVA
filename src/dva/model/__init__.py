from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dva.model.storage_dispatch import (
        StorageDispatchEvaluation,
        StorageDispatchModel,
        StorageDispatchParameters,
        StorageDispatchResult,
        build_storage_dispatch_model,
        evaluate_storage_dispatch_result,
        solve_storage_dispatch,
        solve_storage_dispatch_lexicographic,
    )
    from dva.model.train import (
        DEFAULT_DATASET_PATH,
        DEFAULT_DATE_COLUMN,
        DEFAULT_FEATURE_COLUMNS,
        DEFAULT_HOLDOUT_DAYS,
        DEFAULT_MODEL_NAME,
        DEFAULT_TARGET_COLUMNS,
        DEFAULT_XGB_COLSAMPLE_BYTREE,
        DEFAULT_XGB_LEARNING_RATE,
        DEFAULT_XGB_MAX_DEPTH,
        DEFAULT_XGB_N_ESTIMATORS,
        DEFAULT_XGB_REG_LAMBDA,
        DEFAULT_XGB_SUBSAMPLE,
        DEFAULT_XGB_VERBOSITY,
        ModelTrainingArtifacts,
        NumpyXGBRegressor,
        SUPPORTED_MODEL_NAMES,
        TrainExplainSplit,
        load_default_train_explain_split,
        load_default_training_data,
        load_default_training_frame,
        train_default_model,
        train_model,
    )


__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_DATE_COLUMN",
    "DEFAULT_FEATURE_COLUMNS",
    "DEFAULT_HOLDOUT_DAYS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TARGET_COLUMNS",
    "DEFAULT_XGB_COLSAMPLE_BYTREE",
    "DEFAULT_XGB_LEARNING_RATE",
    "DEFAULT_XGB_MAX_DEPTH",
    "DEFAULT_XGB_N_ESTIMATORS",
    "DEFAULT_XGB_REG_LAMBDA",
    "DEFAULT_XGB_SUBSAMPLE",
    "DEFAULT_XGB_VERBOSITY",
    "ModelTrainingArtifacts",
    "NumpyXGBRegressor",
    "StorageDispatchEvaluation",
    "StorageDispatchModel",
    "StorageDispatchParameters",
    "StorageDispatchResult",
    "SUPPORTED_MODEL_NAMES",
    "TrainExplainSplit",
    "build_storage_dispatch_model",
    "evaluate_storage_dispatch_result",
    "load_default_train_explain_split",
    "load_default_training_data",
    "load_default_training_frame",
    "solve_storage_dispatch",
    "solve_storage_dispatch_lexicographic",
    "train_default_model",
    "train_model",
]

_STORAGE_EXPORTS = {
    "StorageDispatchEvaluation",
    "StorageDispatchModel",
    "StorageDispatchParameters",
    "StorageDispatchResult",
    "build_storage_dispatch_model",
    "evaluate_storage_dispatch_result",
    "solve_storage_dispatch",
    "solve_storage_dispatch_lexicographic",
}
_TRAIN_EXPORTS = {
    "DEFAULT_DATASET_PATH",
    "DEFAULT_DATE_COLUMN",
    "DEFAULT_FEATURE_COLUMNS",
    "DEFAULT_HOLDOUT_DAYS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TARGET_COLUMNS",
    "DEFAULT_XGB_COLSAMPLE_BYTREE",
    "DEFAULT_XGB_LEARNING_RATE",
    "DEFAULT_XGB_MAX_DEPTH",
    "DEFAULT_XGB_N_ESTIMATORS",
    "DEFAULT_XGB_REG_LAMBDA",
    "DEFAULT_XGB_SUBSAMPLE",
    "DEFAULT_XGB_VERBOSITY",
    "ModelTrainingArtifacts",
    "NumpyXGBRegressor",
    "SUPPORTED_MODEL_NAMES",
    "TrainExplainSplit",
    "load_default_train_explain_split",
    "load_default_training_data",
    "load_default_training_frame",
    "train_default_model",
    "train_model",
}


def __getattr__(name: str) -> object:
    if name in _STORAGE_EXPORTS:
        module = import_module("dva.model.storage_dispatch")
        return getattr(module, name)
    if name in _TRAIN_EXPORTS:
        module = import_module("dva.model.train")
        return getattr(module, name)
    raise AttributeError(f"module 'dva.model' has no attribute {name!r}")
