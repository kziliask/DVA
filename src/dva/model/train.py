from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from dva.model.storage_dispatch import (
    StorageDispatchParameters,
    StorageDispatchSPOModel,
    build_spo_training_targets,
)


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
# The cleaned dataset includes an hour-25 column for DST bookkeeping, but the
# default forecasting target here is the next-day 24-hour LMP vector.
DEFAULT_TARGET_COLUMNS = tuple(f"lmp_opr_hour_{hour:02d}" for hour in range(1, 25))
# The expanded CAISO dataset starts on 2023-01-26. After dropping DST rows with
# incomplete 24-hour targets, 101 holdout rows leaves a 36-month training window.
DEFAULT_HOLDOUT_DAYS = 101
DEFAULT_MODEL_NAME = "mlp"
SUPPORTED_MODEL_NAMES = ("rf", "mlp", "torch_mlp", "spo_mlp", "ridge", "svr", "xgb")
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


RandomForestTrainingArtifacts = ModelTrainingArtifacts
RandomForestTrainExplainSplit = TrainExplainSplit


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


class TorchMLPRegressor:
    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...],
        max_iter: int,
        random_state: int | None,
        learning_rate: float = DEFAULT_TORCH_LEARNING_RATE,
        dropout: float = DEFAULT_TORCH_MLP_DROPOUT,
        weight_decay: float = DEFAULT_TORCH_MLP_WEIGHT_DECAY,
        batch_size: int | None = DEFAULT_TORCH_MLP_BATCH_SIZE,
        early_stopping_patience: int | None = DEFAULT_TORCH_MLP_EARLY_STOPPING_PATIENCE,
        activation: str = DEFAULT_TORCH_MLP_ACTIVATION,
        batch_norm: bool = DEFAULT_TORCH_MLP_BATCH_NORM,
        training_verbose: bool = False,
        training_log_every: int | None = None,
        model_label: str = "torch_mlp",
    ) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be strictly positive.")
        if any(hidden_units <= 0 for hidden_units in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain only positive integers.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be strictly positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be strictly positive when provided.")
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience must be strictly positive when provided."
            )
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be one of: relu, gelu.")
        if training_log_every is not None and training_log_every <= 0:
            raise ValueError("training_log_every must be strictly positive when provided.")

        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.activation = activation
        self.batch_norm = batch_norm
        self.training_verbose = training_verbose
        self.training_log_every = training_log_every
        self.model_label = model_label
        self.scaler = StandardScaler()
        self.model_: Any | None = None
        self.output_count_: int | None = None
        self.epochs_trained_: int | None = None
        self.best_validation_loss_: float | None = None

    def _log(self, message: str) -> None:
        if self.training_verbose:
            print(f"[{self.model_label}] {message}", flush=True)

    def _resolved_training_log_every(self) -> int:
        return (
            self.training_log_every
            if self.training_log_every is not None
            else max(1, self.max_iter // 10)
        )

    def _should_log_epoch(self, epoch: int) -> bool:
        if not self.training_verbose:
            return False
        if epoch == 1 or epoch == self.max_iter:
            return True
        return epoch % self._resolved_training_log_every() == 0

    def _prepare_training_arrays(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_array = np.asarray(X_train, dtype=np.float32)
        y_array = np.asarray(y_train, dtype=np.float32)
        if X_array.ndim != 2:
            raise ValueError("X_train must be a 2D feature matrix.")
        if y_array.ndim == 1:
            y_array = y_array[:, np.newaxis]
        elif y_array.ndim != 2:
            raise ValueError("y_train must be a 1D or 2D target array.")
        return X_array, y_array

    def _build_network(self, input_width: int, output_count: int) -> Any:
        from torch import nn

        layers: list[Any] = []
        for hidden_units in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_width, hidden_units))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU() if self.activation == "relu" else nn.GELU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            input_width = hidden_units
        layers.append(nn.Linear(input_width, output_count))
        return nn.Sequential(*layers)

    def _train_validation_split_indices(
        self,
        row_count: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        all_indices = np.arange(row_count, dtype=int)
        if self.early_stopping_patience is None or row_count < 2:
            return all_indices, None
        validation_count = min(max(1, int(np.ceil(row_count * 0.1))), row_count - 1)
        train_count = row_count - validation_count
        return all_indices[:train_count], all_indices[train_count:]

    def _resolved_batch_size(self, train_count: int) -> int:
        if self.batch_size is None:
            return train_count
        return min(self.batch_size, train_count)

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.DataFrame | np.ndarray,
    ) -> TorchMLPRegressor:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                "torch is required for model_name='torch_mlp'. "
                "Install it with `uv add torch`."
            ) from exc

        X_array, y_array = self._prepare_training_arrays(X_train, y_train)
        torch.set_num_threads(1)
        X_scaled = self.scaler.fit_transform(X_array).astype(np.float32, copy=False)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        model = self._build_network(
            input_width=int(X_scaled.shape[1]),
            output_count=int(y_array.shape[1]),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()
        train_indices, validation_indices = self._train_validation_split_indices(
            len(X_scaled),
        )
        X_tensor = torch.from_numpy(X_scaled)
        y_tensor = torch.from_numpy(y_array)
        train_index_tensor = torch.as_tensor(train_indices, dtype=torch.long)
        validation_index_tensor = (
            None
            if validation_indices is None
            else torch.as_tensor(validation_indices, dtype=torch.long)
        )
        batch_size = self._resolved_batch_size(len(train_indices))
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)

        self._log(
            "starting training "
            f"(rows={len(X_array)}, features={X_array.shape[1]}, "
            f"outputs={y_array.shape[1]}, epochs={self.max_iter}, "
            f"learning_rate={self.learning_rate}, dropout={self.dropout}, "
            f"weight_decay={self.weight_decay}, batch_size={batch_size}, "
            f"early_stopping_patience={self.early_stopping_patience}, "
            f"activation={self.activation}, batch_norm={self.batch_norm})"
        )
        training_started_at = time.perf_counter()
        best_validation_loss = math.inf
        best_state_dict: dict[str, Any] | None = None
        epochs_without_improvement = 0
        epochs_trained = 0
        for epoch in range(1, self.max_iter + 1):
            epoch_started_at = time.perf_counter()
            model.train()
            shuffled = train_index_tensor[
                torch.randperm(len(train_index_tensor), generator=generator)
            ]
            epoch_loss_sum = 0.0
            epoch_row_count = 0
            for start in range(0, len(shuffled), batch_size):
                batch_indices = shuffled[start : start + batch_size]
                optimizer.zero_grad()
                predictions = model(X_tensor[batch_indices])
                loss = loss_fn(predictions, y_tensor[batch_indices])
                loss.backward()
                optimizer.step()
                batch_count = int(len(batch_indices))
                epoch_loss_sum += float(loss.detach().item()) * batch_count
                epoch_row_count += batch_count
            epoch_loss = epoch_loss_sum / epoch_row_count
            validation_loss = None
            if validation_index_tensor is not None:
                model.eval()
                with torch.no_grad():
                    validation_predictions = model(X_tensor[validation_index_tensor])
                    validation_loss = float(
                        loss_fn(
                            validation_predictions,
                            y_tensor[validation_index_tensor],
                        )
                        .detach()
                        .item()
                    )
                if validation_loss < best_validation_loss - 1e-12:
                    best_validation_loss = validation_loss
                    best_state_dict = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            epochs_trained = epoch
            if self._should_log_epoch(epoch):
                validation_fragment = (
                    ""
                    if validation_loss is None
                    else f", validation_loss={validation_loss:.6f}"
                )
                self._log(
                    f"epoch {epoch}/{self.max_iter}: "
                    f"loss={epoch_loss:.6f}{validation_fragment}, "
                    f"epoch_seconds={time.perf_counter() - epoch_started_at:.2f}, "
                    f"elapsed_seconds={time.perf_counter() - training_started_at:.2f}"
                )
            if (
                self.early_stopping_patience is not None
                and validation_index_tensor is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                self._log(
                    "early stopping triggered "
                    f"after {epoch} epochs; best_validation_loss={best_validation_loss:.6f}"
                )
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        self.model_ = model.eval()
        self.output_count_ = int(y_array.shape[1])
        self.epochs_trained_ = epochs_trained
        self.best_validation_loss_ = (
            None if math.isinf(best_validation_loss) else float(best_validation_loss)
        )
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        if self.model_ is None or self.output_count_ is None:
            raise RuntimeError("TorchMLPRegressor must be fitted before predict is called.")

        import torch

        X_array = np.asarray(X, dtype=np.float32)
        if X_array.ndim == 1:
            X_array = X_array[np.newaxis, :]
        elif X_array.ndim != 2:
            raise ValueError("X must be a 1D or 2D feature array.")
        X_scaled = self.scaler.transform(X_array).astype(np.float32, copy=False)
        X_tensor = torch.from_numpy(X_scaled)
        with torch.no_grad():
            predictions = self.model_(X_tensor).cpu().numpy()

        if self.output_count_ == 1:
            return predictions[:, 0]
        return predictions


class SPOMLPRegressor(TorchMLPRegressor):
    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...],
        max_iter: int,
        random_state: int | None,
        storage_parameters: StorageDispatchParameters,
        learning_rate: float = DEFAULT_TORCH_LEARNING_RATE,
        mse_learning_rate: float | None = None,
        spo_learning_rate: float | None = None,
        training_verbose: bool = False,
        training_log_every: int | None = None,
        spo_processes: int | None = None,
        warm_start_with_mse: bool = False,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate=learning_rate,
            training_verbose=training_verbose,
            training_log_every=training_log_every,
            model_label="spo_mlp",
        )
        if mse_learning_rate is not None and mse_learning_rate <= 0:
            raise ValueError("mse_learning_rate must be strictly positive when provided.")
        if spo_learning_rate is not None and spo_learning_rate <= 0:
            raise ValueError("spo_learning_rate must be strictly positive when provided.")
        self.storage_parameters = storage_parameters
        self.mse_learning_rate = mse_learning_rate
        self.spo_learning_rate = spo_learning_rate
        self.spo_processes = spo_processes
        self.warm_start_with_mse = warm_start_with_mse

    def _resolved_spo_processes(self) -> int:
        if self.spo_processes is None:
            return max(1, min(4, os.cpu_count() or 1))
        if self.spo_processes < 0:
            raise ValueError("spo_processes must be non-negative when provided.")
        return self.spo_processes

    def _resolved_mse_learning_rate(self) -> float:
        if self.mse_learning_rate is not None:
            return self.mse_learning_rate
        return self.learning_rate

    def _resolved_spo_learning_rate(self) -> float:
        if self.spo_learning_rate is not None:
            return self.spo_learning_rate
        return self.learning_rate

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.DataFrame | np.ndarray,
    ) -> SPOMLPRegressor:
        try:
            import pyepo.func as pyepo_func
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch and pyepo are required for model_name='spo_mlp'. "
                "Install them with `uv add torch pyepo`."
            ) from exc

        X_array, y_array = self._prepare_training_arrays(X_train, y_train)
        torch.set_num_threads(1)
        resolved_mse_learning_rate = self._resolved_mse_learning_rate()
        resolved_spo_learning_rate = self._resolved_spo_learning_rate()
        if self.warm_start_with_mse:
            self._log(
                "MSE warm-start enabled; running an MSE pretraining phase "
                f"for {self.max_iter} epochs before SPO+ fine-tuning"
            )
            original_learning_rate = self.learning_rate
            try:
                self.learning_rate = resolved_mse_learning_rate
                super().fit(X_array, y_array)
                if self.model_ is None:
                    raise RuntimeError("MSE warm-start did not produce a fitted model.")
                model = self.model_
                X_scaled = self.scaler.transform(X_array).astype(np.float32, copy=False)
            finally:
                self.learning_rate = original_learning_rate
        else:
            X_scaled = self.scaler.fit_transform(X_array).astype(np.float32, copy=False)
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            model = self._build_network(
                input_width=int(X_scaled.shape[1]),
                output_count=int(y_array.shape[1]),
            )

        resolved_spo_processes = self._resolved_spo_processes()
        solver_params = {
            "Threads": 1,
            "Seed": 0 if self.random_state is None else self.random_state,
            "MIPGap": 0.0,
            "MIPGapAbs": 1e-9,
        }
        true_costs, true_solutions, true_objectives = build_spo_training_targets(
            y_array,
            self.storage_parameters,
            solver_params=solver_params,
            verbose=self.training_verbose,
            log_every=self.training_log_every,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=resolved_spo_learning_rate)
        loss_fn = pyepo_func.SPOPlus(
            optmodel=StorageDispatchSPOModel(
                parameters=self.storage_parameters,
                horizon=int(y_array.shape[1]),
                solver_params=solver_params,
            ),
            processes=resolved_spo_processes,
        )
        X_tensor = torch.from_numpy(X_scaled)
        true_cost_tensor = torch.from_numpy(true_costs)
        true_solution_tensor = torch.from_numpy(true_solutions)
        true_objective_tensor = torch.from_numpy(true_objectives)
        throughput_penalty = torch.tensor(
            self.storage_parameters.throughput_penalty,
            dtype=torch.float32,
        )

        self._log(
            "starting SPO+ training "
            f"(rows={len(X_array)}, features={X_array.shape[1]}, "
            f"outputs={y_array.shape[1]}, epochs={self.max_iter}, "
            f"spo_processes={resolved_spo_processes}, solver_threads=1, "
            f"warm_start_with_mse={self.warm_start_with_mse}, "
            f"mse_learning_rate={resolved_mse_learning_rate}, "
            f"spo_learning_rate={resolved_spo_learning_rate})"
        )
        model.train()
        training_started_at = time.perf_counter()
        for epoch in range(1, self.max_iter + 1):
            epoch_started_at = time.perf_counter()
            optimizer.zero_grad()
            pred_prices = model(X_tensor)
            loss_started_at = time.perf_counter()
            pred_costs = torch.cat(
                (
                    -(pred_prices + throughput_penalty),
                    pred_prices - throughput_penalty,
                ),
                dim=1,
            )
            loss = loss_fn(
                pred_costs,
                true_cost_tensor,
                true_solution_tensor,
                true_objective_tensor,
            )
            loss.backward()
            optimizer.step()
            if self._should_log_epoch(epoch):
                self._log(
                    f"epoch {epoch}/{self.max_iter}: "
                    f"loss={float(loss.detach().item()):.6f}, "
                    f"spo_loss_seconds={time.perf_counter() - loss_started_at:.2f}, "
                    f"epoch_seconds={time.perf_counter() - epoch_started_at:.2f}, "
                    f"elapsed_seconds={time.perf_counter() - training_started_at:.2f}"
                )

        self.model_ = model.eval()
        self.output_count_ = int(y_array.shape[1])
        return self


def load_default_training_data(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(dataset_path)

    required_columns = [*feature_columns, *target_columns]
    missing_columns = sorted(set(required_columns) - set(dataset.columns))
    if missing_columns:
        raise KeyError(
            "The training dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    training_frame = dataset.loc[:, required_columns].dropna().copy()
    if training_frame.empty:
        raise ValueError("No complete rows remain after filtering the training dataset.")

    X_train = training_frame.loc[:, list(feature_columns)]
    y_train = training_frame.loc[:, list(target_columns)]
    return X_train, y_train


def load_default_random_forest_training_data(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_default_training_data(
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        target_columns=target_columns,
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
    training_frame = training_frame.sort_values(date_column).reset_index(drop=True)
    return training_frame


def load_default_random_forest_training_frame(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    date_column: str = DEFAULT_DATE_COLUMN,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> pd.DataFrame:
    return load_default_training_frame(
        dataset_path=dataset_path,
        date_column=date_column,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )


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

    X_train = train_frame.loc[:, list(feature_columns)]
    y_train = train_frame.loc[:, list(target_columns)]
    X_explain = explain_frame.loc[:, list(feature_columns)]
    y_explain = explain_frame.loc[:, list(target_columns)]

    dataset_path_obj = Path(dataset_path)
    return TrainExplainSplit(
        dataset_path=dataset_path_obj,
        date_column=date_column,
        feature_columns=feature_columns,
        target_columns=target_columns,
        train_frame=train_frame,
        explain_frame=explain_frame,
        train_dates=train_frame.loc[:, date_column].copy(),
        explain_dates=explain_frame.loc[:, date_column].copy(),
        X_train=X_train,
        y_train=y_train,
        X_explain=X_explain,
        y_explain=y_explain,
    )


def load_default_random_forest_train_explain_split(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    date_column: str = DEFAULT_DATE_COLUMN,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> RandomForestTrainExplainSplit:
    return load_default_train_explain_split(
        dataset_path=dataset_path,
        holdout_days=holdout_days,
        date_column=date_column,
        feature_columns=feature_columns,
        target_columns=target_columns,
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
    mlp_hidden_layer_sizes: tuple[int, ...] = DEFAULT_MLP_HIDDEN_LAYER_SIZES,
    mlp_max_iter: int = DEFAULT_MLP_MAX_ITER,
    mlp_dropout: float = DEFAULT_TORCH_MLP_DROPOUT,
    mlp_weight_decay: float = DEFAULT_TORCH_MLP_WEIGHT_DECAY,
    mlp_batch_size: int | None = DEFAULT_TORCH_MLP_BATCH_SIZE,
    mlp_early_stopping_patience: int | None = DEFAULT_TORCH_MLP_EARLY_STOPPING_PATIENCE,
    mlp_activation: str = DEFAULT_TORCH_MLP_ACTIVATION,
    mlp_batch_norm: bool = DEFAULT_TORCH_MLP_BATCH_NORM,
    learning_rate: float | None = None,
    mse_learning_rate: float | None = None,
    spo_learning_rate: float | None = None,
    storage_parameters: StorageDispatchParameters | None = None,
    training_verbose: bool = False,
    training_log_every: int | None = None,
    spo_processes: int | None = None,
    spo_warm_start_with_mse: bool = False,
    xgb_n_estimators: int = DEFAULT_XGB_N_ESTIMATORS,
    xgb_max_depth: int = DEFAULT_XGB_MAX_DEPTH,
    xgb_learning_rate: float = DEFAULT_XGB_LEARNING_RATE,
    xgb_subsample: float = DEFAULT_XGB_SUBSAMPLE,
    xgb_colsample_bytree: float = DEFAULT_XGB_COLSAMPLE_BYTREE,
    xgb_reg_lambda: float = DEFAULT_XGB_REG_LAMBDA,
    xgb_verbosity: int = DEFAULT_XGB_VERBOSITY,
) -> ModelTrainingArtifacts:
    if X_train.empty or y_train.empty:
        raise ValueError("X_train and y_train must both contain at least one row.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            "model_name must be one of: " + ", ".join(SUPPORTED_MODEL_NAMES)
        )

    resolved_feature_columns = (
        feature_columns if feature_columns is not None else tuple(X_train.columns)
    )
    resolved_target_columns = (
        target_columns if target_columns is not None else tuple(y_train.columns)
    )

    if model_name == "rf":
        model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
        model_description = "RandomForestRegressor"
    elif model_name == "xgb":
        xgb_params = _build_xgb_params(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            reg_lambda=xgb_reg_lambda,
            random_state=random_state,
            verbosity=xgb_verbosity,
        )
        model = NumpyXGBRegressor(
            params=xgb_params,
            feature_columns=resolved_feature_columns,
        )
        model_description = (
            "XGBRegressor"
            f"(n_estimators={xgb_n_estimators}, max_depth={xgb_max_depth}, "
            f"learning_rate={xgb_learning_rate}, subsample={xgb_subsample}, "
            f"colsample_bytree={xgb_colsample_bytree}, reg_lambda={xgb_reg_lambda})"
        )
    elif model_name == "mlp":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=mlp_hidden_layer_sizes,
                        activation="relu",
                        solver="adam",
                        random_state=random_state,
                        max_iter=mlp_max_iter,
                    ),
                ),
            ]
        )
        model_description = (
            "StandardScaler+MLPRegressor"
            f"(hidden_layer_sizes={mlp_hidden_layer_sizes}, max_iter={mlp_max_iter})"
        )
    elif model_name == "torch_mlp":
        model = TorchMLPRegressor(
            hidden_layer_sizes=mlp_hidden_layer_sizes,
            max_iter=mlp_max_iter,
            random_state=random_state,
            learning_rate=(
                DEFAULT_TORCH_LEARNING_RATE if learning_rate is None else learning_rate
            ),
            dropout=mlp_dropout,
            weight_decay=mlp_weight_decay,
            batch_size=mlp_batch_size,
            early_stopping_patience=mlp_early_stopping_patience,
            activation=mlp_activation,
            batch_norm=mlp_batch_norm,
            training_verbose=training_verbose,
            training_log_every=training_log_every,
            model_label="torch_mlp",
        )
        model_description = (
            "StandardScaler+TorchMLPRegressor"
            f"(hidden_layer_sizes={mlp_hidden_layer_sizes}, max_iter={mlp_max_iter}, "
            f"learning_rate={DEFAULT_TORCH_LEARNING_RATE if learning_rate is None else learning_rate}, "
            f"dropout={mlp_dropout}, weight_decay={mlp_weight_decay}, "
            f"batch_size={mlp_batch_size}, "
            f"early_stopping_patience={mlp_early_stopping_patience}, "
            f"activation={mlp_activation}, batch_norm={mlp_batch_norm})"
        )
    elif model_name == "spo_mlp":
        if storage_parameters is None:
            raise ValueError(
                "storage_parameters is required when model_name='spo_mlp'."
            )
        model = SPOMLPRegressor(
            hidden_layer_sizes=mlp_hidden_layer_sizes,
            max_iter=mlp_max_iter,
            random_state=random_state,
            storage_parameters=storage_parameters,
            learning_rate=(
                DEFAULT_TORCH_LEARNING_RATE if learning_rate is None else learning_rate
            ),
            mse_learning_rate=mse_learning_rate,
            spo_learning_rate=spo_learning_rate,
            training_verbose=training_verbose,
            training_log_every=training_log_every,
            spo_processes=spo_processes,
            warm_start_with_mse=spo_warm_start_with_mse,
        )
        model_description = (
            "StandardScaler+SPOMLPRegressor"
            f"(hidden_layer_sizes={mlp_hidden_layer_sizes}, max_iter={mlp_max_iter}, "
            f"spo_processes={'auto' if spo_processes is None else spo_processes}, "
            f"warm_start_with_mse={spo_warm_start_with_mse}, "
            f"learning_rate={DEFAULT_TORCH_LEARNING_RATE if learning_rate is None else learning_rate}, "
            f"mse_learning_rate={mse_learning_rate}, spo_learning_rate={spo_learning_rate})"
        )
    elif model_name == "ridge":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge()),
            ]
        )
        model_description = "StandardScaler+Ridge"
    else:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    # sklearn's SVR is single-target, so wrap it for the 24-hour output vector.
                    "regressor",
                    MultiOutputRegressor(SVR(), n_jobs=n_jobs),
                ),
            ]
        )
        model_description = "StandardScaler+MultiOutputRegressor(SVR)"
    model.fit(X_train, y_train)

    return ModelTrainingArtifacts(
        model=model,
        model_name=model_name,
        model_description=model_description,
        feature_columns=resolved_feature_columns,
        target_columns=resolved_target_columns,
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
        "n_jobs": 1,
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


def train_random_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | None = None,
    target_columns: tuple[str, ...] | None = None,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> RandomForestTrainingArtifacts:
    return train_model(
        X_train,
        y_train,
        model_name="rf",
        feature_columns=feature_columns,
        target_columns=target_columns,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def train_default_random_forest_model(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    *,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> RandomForestTrainingArtifacts:
    X_train, y_train = load_default_training_data(dataset_path=dataset_path)
    return train_model(
        X_train,
        y_train,
        model_name="rf",
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        target_columns=DEFAULT_TARGET_COLUMNS,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def main() -> None:
    artifacts = train_default_random_forest_model()
    print(
        "Trained RandomForestRegressor "
        f"on {len(artifacts.X_train)} rows "
        f"with {len(artifacts.feature_columns)} inputs "
        f"and {len(artifacts.target_columns)} hourly targets."
    )


if __name__ == "__main__":
    main()
