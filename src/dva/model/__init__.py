from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dva.model.orienteering import (
        DEFAULT_MAX_DISTANCE_BUDGET,
        DEFAULT_START_ZONE_ID,
        DEFAULT_ZONE_DISTANCE_MATRIX_PATH,
        OrienteeringModel,
        OrienteeringSolveMethod,
        OrienteeringResult,
        build_orienteering_model,
        load_zone_distance_matrix,
        solve_orienteering,
        solve_orienteering_heuristic,
        solve_orienteering_ortools,
    )
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
        DEFAULT_MLP_HIDDEN_LAYER_SIZES,
        DEFAULT_MLP_MAX_ITER,
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
        RandomForestTrainExplainSplit,
        RandomForestTrainingArtifacts,
        SUPPORTED_MODEL_NAMES,
        TrainExplainSplit,
        load_default_train_explain_split,
        load_default_training_data,
        load_default_random_forest_train_explain_split,
        load_default_training_frame,
        load_default_random_forest_training_frame,
        load_default_random_forest_training_data,
        train_model,
        train_random_forest_model,
        train_default_random_forest_model,
    )


__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_DATE_COLUMN",
    "DEFAULT_FEATURE_COLUMNS",
    "DEFAULT_HOLDOUT_DAYS",
    "DEFAULT_MLP_HIDDEN_LAYER_SIZES",
    "DEFAULT_MLP_MAX_ITER",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MAX_DISTANCE_BUDGET",
    "DEFAULT_START_ZONE_ID",
    "DEFAULT_TARGET_COLUMNS",
    "DEFAULT_XGB_COLSAMPLE_BYTREE",
    "DEFAULT_XGB_LEARNING_RATE",
    "DEFAULT_XGB_MAX_DEPTH",
    "DEFAULT_XGB_N_ESTIMATORS",
    "DEFAULT_XGB_REG_LAMBDA",
    "DEFAULT_XGB_SUBSAMPLE",
    "DEFAULT_XGB_VERBOSITY",
    "DEFAULT_ZONE_DISTANCE_MATRIX_PATH",
    "ModelTrainingArtifacts",
    "OrienteeringModel",
    "OrienteeringSolveMethod",
    "OrienteeringResult",
    "RandomForestTrainExplainSplit",
    "RandomForestTrainingArtifacts",
    "StorageDispatchEvaluation",
    "StorageDispatchModel",
    "StorageDispatchParameters",
    "StorageDispatchResult",
    "build_orienteering_model",
    "build_storage_dispatch_model",
    "evaluate_storage_dispatch_result",
    "load_zone_distance_matrix",
    "load_default_train_explain_split",
    "load_default_training_data",
    "load_default_training_frame",
    "load_default_random_forest_train_explain_split",
    "load_default_random_forest_training_frame",
    "load_default_random_forest_training_data",
    "solve_orienteering",
    "solve_orienteering_heuristic",
    "solve_orienteering_ortools",
    "solve_storage_dispatch",
    "solve_storage_dispatch_lexicographic",
    "SUPPORTED_MODEL_NAMES",
    "TrainExplainSplit",
    "train_model",
    "train_random_forest_model",
    "train_default_random_forest_model",
]

_ORIENTEERING_EXPORTS = {
    "DEFAULT_MAX_DISTANCE_BUDGET",
    "DEFAULT_START_ZONE_ID",
    "DEFAULT_ZONE_DISTANCE_MATRIX_PATH",
    "OrienteeringModel",
    "OrienteeringSolveMethod",
    "OrienteeringResult",
    "build_orienteering_model",
    "load_zone_distance_matrix",
    "solve_orienteering",
    "solve_orienteering_heuristic",
    "solve_orienteering_ortools",
}
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
    "DEFAULT_MLP_HIDDEN_LAYER_SIZES",
    "DEFAULT_MLP_MAX_ITER",
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
    "RandomForestTrainExplainSplit",
    "RandomForestTrainingArtifacts",
    "SUPPORTED_MODEL_NAMES",
    "TrainExplainSplit",
    "load_default_train_explain_split",
    "load_default_training_data",
    "load_default_training_frame",
    "load_default_random_forest_train_explain_split",
    "load_default_random_forest_training_frame",
    "load_default_random_forest_training_data",
    "train_model",
    "train_random_forest_model",
    "train_default_random_forest_model",
}


def __getattr__(name: str) -> object:
    if name in _ORIENTEERING_EXPORTS:
        module = import_module("dva.model.orienteering")
        return getattr(module, name)
    if name in _STORAGE_EXPORTS:
        module = import_module("dva.model.storage_dispatch")
        return getattr(module, name)
    if name in _TRAIN_EXPORTS:
        module = import_module("dva.model.train")
        return getattr(module, name)
    raise AttributeError(f"module 'dva.model' has no attribute {name!r}")
