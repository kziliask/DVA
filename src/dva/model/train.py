from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


DEFAULT_DATASET_PATH = Path("data/cleaned/caiso_sp15_daily_lmp_weather_2023-01-26_2026-05-07.csv")
DEFAULT_DATE_COLUMN = "date"
WEATHER_FEATURE_COLUMNS = (
    "min_temp_c",
    "max_temp_c",
    "mean_temp_c",
    "mean_humidity",
    "mean_wind_speed",
    "mean_solar_irradiance",
    "max_solar_irradiance",
)
DEFAULT_FEATURE_COLUMNS = (*WEATHER_FEATURE_COLUMNS, "day_of_week")
DEFAULT_TARGET_COLUMNS = tuple(f"lmp_opr_hour_{hour:02d}" for hour in range(1, 25))
DEFAULT_HOLDOUT_DAYS = 101
DEFAULT_MODEL_NAME = "xgb"
SUPPORTED_MODEL_NAMES = ("xgb",)
DEFAULT_MLP_HIDDEN_LAYER_SIZES = (256,)
DEFAULT_MLP_MAX_ITER = 100
DEFAULT_TORCH_LEARNING_RATE = 1e-3
DEFAULT_TORCH_MLP_DROPOUT = 0.0
DEFAULT_TORCH_MLP_WEIGHT_DECAY = 0.0
DEFAULT_TORCH_MLP_BATCH_SIZE: int | None = None
DEFAULT_TORCH_MLP_EARLY_STOPPING_PATIENCE: int | None = None
DEFAULT_TORCH_MLP_ACTIVATION = "relu"
DEFAULT_TORCH_MLP_BATCH_NORM = False
DEFAULT_XGB_N_ESTIMATORS = 100
DEFAULT_XGB_MAX_DEPTH = 3
DEFAULT_XGB_LEARNING_RATE = 0.05
DEFAULT_XGB_SUBSAMPLE = 0.9
DEFAULT_XGB_COLSAMPLE_BYTREE = 0.9
DEFAULT_XGB_REG_LAMBDA = 1.0
DEFAULT_XGB_VERBOSITY = 0


@dataclass(frozen=True, slots=True)
class ModelTrainingArtifacts:
    model: Any
    model_name: str
    model_description: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    X_train: pd.DataFrame
    y_train: pd.DataFrame


@dataclass(frozen=True, slots=True)
class TrainExplainSplit:
    dataset_path: Path
    date_column: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    train_frame: pd.DataFrame
    explain_frame: pd.DataFrame
    train_dates: pd.Series
    explain_dates: pd.Series
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_explain: pd.DataFrame
    y_explain: pd.DataFrame


class NumpyXGBRegressor:
    def __init__(
        self,
        *,
        params: dict[str, Any],
        feature_columns: tuple[str, ...],
    ) -> None:
        self.params = dict(params)
        self.feature_columns = tuple(feature_columns)
        self.model = XGBRegressor(**self.params)

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.DataFrame | np.ndarray,
    ) -> NumpyXGBRegressor:
        self.model.fit(
            self._feature_array(X_train),
            _as_float32_matrix(y_train),
        )
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        return np.asarray(self.model.predict(self._feature_array(X)), dtype=float)

    def _feature_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return _as_float32_matrix(X.loc[:, list(self.feature_columns)])
        return _as_float32_matrix(X)


def load_default_training_data(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_frame = load_default_training_frame(
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )
    return (
        training_frame.loc[:, list(feature_columns)],
        training_frame.loc[:, list(target_columns)],
    )


def load_default_training_frame(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    date_column: str = DEFAULT_DATE_COLUMN,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)

    required_columns = [date_column, *feature_columns, *target_columns]
    missing_columns = sorted(set(required_columns) - set(dataset.columns))
    if missing_columns:
        raise KeyError(
            "The training dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    training_frame = dataset.loc[:, required_columns].dropna().copy()
    if training_frame.empty:
        raise ValueError("No complete rows remain after filtering the training dataset.")

    training_frame[date_column] = training_frame[date_column].astype(str)
    return training_frame.sort_values(date_column).reset_index(drop=True)


def load_default_train_explain_split(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    date_column: str = DEFAULT_DATE_COLUMN,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> TrainExplainSplit:
    training_frame = load_default_training_frame(
        dataset_path=dataset_path,
        date_column=date_column,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )

    if holdout_days <= 0:
        raise ValueError("holdout_days must be strictly positive.")
    if holdout_days >= len(training_frame):
        raise ValueError(
            "holdout_days must be smaller than the number of complete training rows."
        )

    train_frame = training_frame.iloc[:-holdout_days].reset_index(drop=True)
    explain_frame = training_frame.iloc[-holdout_days:].reset_index(drop=True)

    return TrainExplainSplit(
        dataset_path=Path(dataset_path),
        date_column=date_column,
        feature_columns=tuple(feature_columns),
        target_columns=tuple(target_columns),
        train_frame=train_frame,
        explain_frame=explain_frame,
        train_dates=train_frame.loc[:, date_column].copy(),
        explain_dates=explain_frame.loc[:, date_column].copy(),
        X_train=train_frame.loc[:, list(feature_columns)],
        y_train=train_frame.loc[:, list(target_columns)],
        X_explain=explain_frame.loc[:, list(feature_columns)],
        y_explain=explain_frame.loc[:, list(target_columns)],
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    feature_columns: tuple[str, ...] | None = None,
    target_columns: tuple[str, ...] | None = None,
    random_state: int | None = None,
    n_jobs: int | None = None,
    xgb_n_estimators: int = DEFAULT_XGB_N_ESTIMATORS,
    xgb_max_depth: int = DEFAULT_XGB_MAX_DEPTH,
    xgb_learning_rate: float = DEFAULT_XGB_LEARNING_RATE,
    xgb_subsample: float = DEFAULT_XGB_SUBSAMPLE,
    xgb_colsample_bytree: float = DEFAULT_XGB_COLSAMPLE_BYTREE,
    xgb_reg_lambda: float = DEFAULT_XGB_REG_LAMBDA,
    xgb_verbosity: int = DEFAULT_XGB_VERBOSITY,
    **deprecated_model_kwargs: Any,
) -> ModelTrainingArtifacts:
    if X_train.empty or y_train.empty:
        raise ValueError("X_train and y_train must both contain at least one row.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if model_name != "xgb":
        raise ValueError(
            "Only model_name='xgb' is supported. "
            "The archived legacy implementations are kept locally as train_old.py."
        )

    resolved_feature_columns = (
        feature_columns if feature_columns is not None else tuple(X_train.columns)
    )
    resolved_target_columns = (
        target_columns if target_columns is not None else tuple(y_train.columns)
    )
    xgb_params = _build_xgb_params(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        reg_lambda=xgb_reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=xgb_verbosity,
    )
    model = NumpyXGBRegressor(
        params=xgb_params,
        feature_columns=tuple(resolved_feature_columns),
    )
    model.fit(X_train, y_train)

    return ModelTrainingArtifacts(
        model=model,
        model_name="xgb",
        model_description=(
            "XGBRegressor"
            f"(n_estimators={xgb_n_estimators}, max_depth={xgb_max_depth}, "
            f"learning_rate={xgb_learning_rate}, subsample={xgb_subsample}, "
            f"colsample_bytree={xgb_colsample_bytree}, reg_lambda={xgb_reg_lambda})"
        ),
        feature_columns=tuple(resolved_feature_columns),
        target_columns=tuple(resolved_target_columns),
        X_train=X_train,
        y_train=y_train,
    )


def _build_xgb_params(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    random_state: int | None,
    n_jobs: int | None,
    verbosity: int,
) -> dict[str, Any]:
    if n_estimators <= 0:
        raise ValueError("xgb_n_estimators must be strictly positive.")
    if max_depth <= 0:
        raise ValueError("xgb_max_depth must be strictly positive.")
    if learning_rate <= 0:
        raise ValueError("xgb_learning_rate must be strictly positive.")
    if not 0.0 < subsample <= 1.0:
        raise ValueError("xgb_subsample must be in (0, 1].")
    if not 0.0 < colsample_bytree <= 1.0:
        raise ValueError("xgb_colsample_bytree must be in (0, 1].")

    return {
        "objective": "reg:squarederror",
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_lambda": float(reg_lambda),
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": 1 if n_jobs is None else int(n_jobs),
        "verbosity": int(verbosity),
    }


def _as_float32_matrix(values: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        array = values.to_numpy(dtype=np.float32, copy=True)
    else:
        array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    return np.ascontiguousarray(array)


def train_default_model(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> ModelTrainingArtifacts:
    X_train, y_train = load_default_training_data(dataset_path=dataset_path)
    return train_model(
        X_train,
        y_train,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        target_columns=DEFAULT_TARGET_COLUMNS,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def main() -> None:
    artifacts = train_default_model()
    print(
        "Trained XGBRegressor "
        f"on {len(artifacts.X_train)} rows "
        f"with {len(artifacts.feature_columns)} inputs "
        f"and {len(artifacts.target_columns)} hourly targets."
    )


if __name__ == "__main__":
    main()
