from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dva.analysis.evaluation_metrics import (
    DECISION_DELETION_AUC_CLIPPING,
    DECISION_INSERTION_AUC_CLIPPING,
    build_attribution_ranking,
    build_metric_summary,
    compute_decision_activation_metrics,
    compute_decision_deletion_auc,
    compute_decision_insertion_auc,
    compute_exact_decision_infidelity,
    compute_kendall_tau_correlation,
    compute_rank_kendall_tau_from_rankings,
    compute_rank_spearman_from_rankings,
    compute_spearman_rank_correlation,
)
from dva.model.storage_dispatch import (
    StorageDispatchParameters,
    StorageDispatchResult,
    evaluate_storage_dispatch_result,
    solve_storage_dispatch_lexicographic,
)
from dva.model.train import (
    DEFAULT_DATASET_PATH,
    DEFAULT_HOLDOUT_DAYS,
    DEFAULT_XGB_COLSAMPLE_BYTREE,
    DEFAULT_XGB_LEARNING_RATE,
    DEFAULT_XGB_MAX_DEPTH,
    DEFAULT_XGB_N_ESTIMATORS,
    DEFAULT_XGB_REG_LAMBDA,
    DEFAULT_XGB_SUBSAMPLE,
    DEFAULT_XGB_VERBOSITY,
    DEFAULT_MLP_HIDDEN_LAYER_SIZES,
    DEFAULT_MLP_MAX_ITER,
    DEFAULT_TORCH_MLP_ACTIVATION,
    DEFAULT_TORCH_MLP_BATCH_NORM,
    DEFAULT_TORCH_MLP_BATCH_SIZE,
    DEFAULT_TORCH_MLP_DROPOUT,
    DEFAULT_TORCH_MLP_EARLY_STOPPING_PATIENCE,
    DEFAULT_TORCH_MLP_WEIGHT_DECAY,
    ModelTrainingArtifacts,
    SUPPORTED_MODEL_NAMES,
    load_default_train_explain_split,
    train_model,
)


DEFAULT_RANDOM_STATE = 0
DEFAULT_RF_N_JOBS = 1
DEFAULT_SOLVER_SEED = 0
DEFAULT_MIP_GAP = 0.0
DEFAULT_MIP_GAP_ABS = 1e-9
DEFAULT_OBJECTIVE_TOLERANCE = 1e-6
DEFAULT_OUTPUT_DIR = Path("results/caiso_shap_case_study")
DEFAULT_CAISO_MODEL_NAME = "xgb"
SUPPORTED_INTERACTION_METHODS = ("shapley_taylor", "faith_shap")
DEFAULT_INTERACTION_METHOD = "shapley_taylor"
DEFAULT_BACKGROUND_DAYS = 365
CAISO_DECISION_ACTIVATION_DISPATCH_ATOL = 1e-6


@dataclass(frozen=True, slots=True)
class DailyShapExplanation:
    date: str
    predictive_hourly_shap: np.ndarray
    predictive_daily_shap: np.ndarray
    decision_shap: np.ndarray
    ead_decision_shap: np.ndarray | None
    decision_activation_rate: np.ndarray
    decision_activated_value_sum: np.ndarray
    decision_activated_value: np.ndarray
    evaluation_metrics_by_family: dict[str, dict[str, float | None]]
    predictive_baseline_total: float
    predictive_full_total: float
    decision_baseline_value: float
    decision_full_value: float
    ead_decision_baseline_value: float | None
    ead_decision_full_value: float | None
    ead_decision_characteristic_baseline_value: float | None
    ead_decision_characteristic_full_value: float | None
    ead_decision_value_gain: float | None
    oracle_objective_value: float
    decision_value_gain: float
    daily_abs_rank_spearman: float | None
    daily_abs_rank_kendall_tau: float | None
    predictive_ead_abs_rank_spearman: float | None
    predictive_ead_abs_rank_kendall_tau: float | None
    decision_ead_abs_rank_spearman: float | None
    decision_ead_abs_rank_kendall_tau: float | None


@dataclass(frozen=True, slots=True)
class DailyInteractionExplanation:
    date: str
    method: str
    order: int
    player_names: tuple[str, ...]
    decision_indices: dict[frozenset[int], float]
    predictive_indices: dict[frozenset[int], np.ndarray]
    decision_value_full: float
    predictive_value_full: np.ndarray
    predictive_value_empty: np.ndarray


@dataclass(frozen=True, slots=True)
class DailyShapleyTaylorExplanation:
    date: str
    order: int
    player_names: tuple[str, ...]
    decision_indices: dict[frozenset[int], float]
    predictive_indices: dict[frozenset[int], np.ndarray]
    decision_value_full: float
    predictive_value_full: np.ndarray
    predictive_value_empty: np.ndarray


@dataclass(frozen=True, slots=True)
class CaisoShapCaseStudyOutputs:
    daily_shap: pd.DataFrame
    predictive_hourly_shap: pd.DataFrame
    daily_full_dispatch: pd.DataFrame
    summary_shap: pd.DataFrame
    prediction_metrics: dict[str, Any]
    comparison_metrics: dict[str, Any]
    evaluation_metrics: dict[str, Any]
    run_metadata: dict[str, Any]
    daily_interaction_decision: pd.DataFrame | None = None
    daily_interaction_predictive: pd.DataFrame | None = None
    daily_shapley_taylor_decision: pd.DataFrame | None = None
    daily_shapley_taylor_predictive: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class CaisoShapCaseStudyConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    holdout_days: int = DEFAULT_HOLDOUT_DAYS
    outdir: Path = DEFAULT_OUTPUT_DIR
    model_name: str = DEFAULT_CAISO_MODEL_NAME
    random_state: int = DEFAULT_RANDOM_STATE
    n_jobs: int = DEFAULT_RF_N_JOBS
    mlp_hidden_layer_sizes: tuple[int, ...] = DEFAULT_MLP_HIDDEN_LAYER_SIZES
    mlp_max_iter: int = DEFAULT_MLP_MAX_ITER
    mlp_dropout: float = DEFAULT_TORCH_MLP_DROPOUT
    mlp_weight_decay: float = DEFAULT_TORCH_MLP_WEIGHT_DECAY
    mlp_batch_size: int | None = DEFAULT_TORCH_MLP_BATCH_SIZE
    mlp_early_stopping_patience: int | None = DEFAULT_TORCH_MLP_EARLY_STOPPING_PATIENCE
    mlp_activation: str = DEFAULT_TORCH_MLP_ACTIVATION
    mlp_batch_norm: bool = DEFAULT_TORCH_MLP_BATCH_NORM
    xgb_n_estimators: int = DEFAULT_XGB_N_ESTIMATORS
    xgb_max_depth: int = DEFAULT_XGB_MAX_DEPTH
    xgb_learning_rate: float = DEFAULT_XGB_LEARNING_RATE
    xgb_subsample: float = DEFAULT_XGB_SUBSAMPLE
    xgb_colsample_bytree: float = DEFAULT_XGB_COLSAMPLE_BYTREE
    xgb_reg_lambda: float = DEFAULT_XGB_REG_LAMBDA
    xgb_verbosity: int = DEFAULT_XGB_VERBOSITY
    learning_rate: float | None = None
    mse_learning_rate: float | None = None
    spo_learning_rate: float | None = None
    training_verbose: bool = False
    training_log_every: int | None = None
    spo_processes: int | None = None
    spo_warm_start_with_mse: bool = False
    solver_seed: int = DEFAULT_SOLVER_SEED
    mip_gap: float = DEFAULT_MIP_GAP
    mip_gap_abs: float = DEFAULT_MIP_GAP_ABS
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE
    max_days: int | None = None
    background_days: int = DEFAULT_BACKGROUND_DAYS
    holdout_mean_impute_features: tuple[str, ...] = ()
    compute_ead_decision_shap: bool = False
    storage_parameters: StorageDispatchParameters = field(
        default_factory=lambda: build_default_storage_parameters()
    )
    interaction_order: int | None = None
    interaction_method: str = DEFAULT_INTERACTION_METHOD
    parameter_player_spec: ParameterPlayerSpec | None = None


class ExactRandomForestCoalitionEvaluator:
    def __init__(
        self,
        model: RandomForestRegressor,
        feature_names: Sequence[str],
    ) -> None:
        if not hasattr(model, "estimators_"):
            raise ValueError("model must be a fitted RandomForestRegressor.")

        self.model = model
        self.feature_names = tuple(feature_names)
        self.feature_count = len(self.feature_names)
        self.coalition_count = 1 << self.feature_count
        self.full_mask = self.coalition_count - 1
        self.output_count = int(getattr(model, "n_outputs_", 1))
        model_feature_count = int(getattr(model, "n_features_in_", self.feature_count))
        if model_feature_count != self.feature_count:
            raise ValueError(
                "feature_names length must match model.n_features_in_. "
                f"Got {self.feature_count} names for {model_feature_count} features."
            )

    def evaluate_all_coalitions(
        self,
        observation: pd.Series | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        x = _resolve_observation_array(observation, self.feature_names)
        coalition_predictions = np.zeros(
            (self.coalition_count, self.output_count),
            dtype=float,
        )
        for coalition_mask in range(self.coalition_count):
            coalition_predictions[coalition_mask] = self._predict_for_mask(x, coalition_mask)
        return coalition_predictions

    def _predict_for_mask(self, x: np.ndarray, coalition_mask: int) -> np.ndarray:
        tree_predictions = [
            _evaluate_tree_expectation(estimator.tree_, x, coalition_mask)
            for estimator in self.model.estimators_
        ]
        return np.mean(np.stack(tree_predictions, axis=0), axis=0)


class BackgroundMarginalCoalitionEvaluator:
    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        background_data: pd.DataFrame,
    ) -> None:
        self.model = model
        self.feature_names = tuple(feature_names)
        self.feature_count = len(self.feature_names)
        self.coalition_count = 1 << self.feature_count
        self.background_frame = background_data.loc[:, list(self.feature_names)].copy()
        self.background_matrix = self.background_frame.to_numpy(
            dtype=float,
            copy=True,
        )
        self.output_count = self._resolve_output_count()

    def evaluate_all_coalitions(
        self,
        observation: pd.Series | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        x = _resolve_observation_array(observation, self.feature_names)
        coalition_predictions = np.zeros(
            (self.coalition_count, self.output_count),
            dtype=float,
        )
        for coalition_mask in range(self.coalition_count):
            coalition_predictions[coalition_mask] = self._predict_for_mask(x, coalition_mask)
        return coalition_predictions

    def _predict_for_mask(self, x: np.ndarray, coalition_mask: int) -> np.ndarray:
        completed_background = self.background_matrix.copy()
        for feature_idx in range(self.feature_count):
            if coalition_mask & (1 << feature_idx):
                completed_background[:, feature_idx] = x[feature_idx]

        completed_background_frame = pd.DataFrame(
            completed_background,
            columns=self.feature_names,
        )
        predictions = np.asarray(
            self.model.predict(completed_background_frame),
            dtype=float,
        )
        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
        return predictions.mean(axis=0)

    def _resolve_output_count(self) -> int:
        sample_prediction = np.asarray(
            self.model.predict(self.background_frame.iloc[:1]),
            dtype=float,
        )
        if sample_prediction.ndim == 1:
            return 1
        return int(sample_prediction.shape[1])


class ExtendedPlayerCoalitionEvaluator:
    """Coalition evaluator over (features union parameters).

    Player layout (bit indices, low -> high):
        [feature_0, ..., feature_{F-1}, param_0, ..., param_{P-1}]
    where P = parameter_player_spec.player_count.

    For a full mask m with F + P bits:
        feature_mask = m & ((1 << F) - 1)
        param_mask = m >> F
    """

    def __init__(
        self,
        feature_evaluator: Any,
        feature_names: Sequence[str],
        actual_parameters: StorageDispatchParameters,
        parameter_player_spec: ParameterPlayerSpec,
    ) -> None:
        self.feature_evaluator = feature_evaluator
        self.feature_names = tuple(feature_names)
        self.feature_count = len(self.feature_names)
        self.actual_parameters = actual_parameters
        self.spec = parameter_player_spec
        self.parameter_names = self.spec.player_names()
        self.parameter_count = len(self.parameter_names)
        self.player_count = self.feature_count + self.parameter_count
        self.coalition_count = 1 << self.player_count
        self.output_count = int(getattr(feature_evaluator, "output_count"))

    def player_names(self) -> tuple[str, ...]:
        return self.feature_names + self.parameter_names

    def feature_predictions_for_all_feature_coalitions(
        self,
        observation: pd.Series | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return self.feature_evaluator.evaluate_all_coalitions(observation)

    def parameters_for_param_mask(self, param_mask: int) -> StorageDispatchParameters:
        bit = 0
        kwargs: dict[str, float | None] = {}
        if self.spec.throughput_penalty_is_player:
            kwargs["throughput_penalty"] = (
                self.actual_parameters.throughput_penalty
                if param_mask & (1 << bit)
                else self.spec.throughput_penalty_baseline
            )
            bit += 1
        else:
            kwargs["throughput_penalty"] = self.actual_parameters.throughput_penalty

        if self.spec.efficiency_is_player:
            if param_mask & (1 << bit):
                kwargs["charge_efficiency"] = self.actual_parameters.charge_efficiency
                kwargs["discharge_efficiency"] = self.actual_parameters.discharge_efficiency
            else:
                kwargs["charge_efficiency"] = self.spec.charge_efficiency_baseline
                kwargs["discharge_efficiency"] = self.spec.discharge_efficiency_baseline
            bit += 1
        else:
            kwargs["charge_efficiency"] = self.actual_parameters.charge_efficiency
            kwargs["discharge_efficiency"] = self.actual_parameters.discharge_efficiency

        if self.spec.energy_capacity_is_player:
            kwargs["energy_capacity"] = (
                self.actual_parameters.energy_capacity
                if param_mask & (1 << bit)
                else self.spec.energy_capacity_baseline
            )
            bit += 1
        else:
            kwargs["energy_capacity"] = self.actual_parameters.energy_capacity

        kwargs["power_limit"] = self.actual_parameters.power_limit
        kwargs["initial_state_of_charge"] = self.actual_parameters.initial_state_of_charge
        kwargs["terminal_state_of_charge"] = self.actual_parameters.terminal_state_of_charge
        return StorageDispatchParameters(**kwargs)


def build_default_storage_parameters() -> StorageDispatchParameters:
    return StorageDispatchParameters(
        energy_capacity=2.0,
        power_limit=1.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        throughput_penalty=0.0,
        initial_state_of_charge=1.0,
        terminal_state_of_charge=1.0,
    )


def select_recent_background_frame(
    frame: pd.DataFrame,
    date_column: str,
    background_days: int,
) -> pd.DataFrame:
    if background_days <= 0:
        raise ValueError("background_days must be strictly positive.")
    if frame.empty:
        raise ValueError("Cannot select background rows from an empty frame.")

    dates = pd.to_datetime(frame.loc[:, date_column], errors="raise")
    end_date = dates.max()
    start_date = end_date - pd.Timedelta(days=background_days - 1)
    background_frame = frame.loc[dates >= start_date].reset_index(drop=True)
    if background_frame.empty:
        raise ValueError("No background rows remain after applying background_days.")
    return background_frame


@dataclass(frozen=True, slots=True)
class ParameterPlayerSpec:
    """Parameter-player baselines for a self-consistent decision-value game.

    For each enabled flag, the coalition uses the actual storage parameter value
    when that player is IN and the corresponding baseline when that player is
    OUT. The defaults encode an uninformed planner baseline: no throughput
    penalty, lossless storage, and effectively unbounded energy capacity.
    """

    throughput_penalty_is_player: bool = False
    throughput_penalty_baseline: float = 0.0
    efficiency_is_player: bool = False
    charge_efficiency_baseline: float = 1.0
    discharge_efficiency_baseline: float = 1.0
    energy_capacity_is_player: bool = False
    energy_capacity_baseline: float = 1e6

    @property
    def player_count(self) -> int:
        return (
            int(self.throughput_penalty_is_player)
            + int(self.efficiency_is_player)
            + int(self.energy_capacity_is_player)
        )

    def player_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.throughput_penalty_is_player:
            names.append("throughput_penalty")
        if self.efficiency_is_player:
            names.append("efficiency")
        if self.energy_capacity_is_player:
            names.append("energy_capacity")
        return tuple(names)


def compute_exact_shapley_values(
    coalition_values: np.ndarray,
    feature_count: int,
) -> np.ndarray:
    expected_coalition_count = 1 << feature_count
    coalition_array = np.asarray(coalition_values, dtype=float)
    if coalition_array.shape[0] != expected_coalition_count:
        raise ValueError(
            "coalition_values must have one entry per coalition. "
            f"Expected {expected_coalition_count}, got {coalition_array.shape[0]}."
        )

    shap_values = np.zeros((feature_count, *coalition_array.shape[1:]), dtype=float)
    weights = _subset_weights(feature_count)
    for feature_idx in range(feature_count):
        feature_bit = 1 << feature_idx
        for coalition_mask in range(expected_coalition_count):
            if coalition_mask & feature_bit:
                continue
            coalition_size = coalition_mask.bit_count()
            marginal_contribution = (
                coalition_array[coalition_mask | feature_bit] - coalition_array[coalition_mask]
            )
            shap_values[feature_idx] += weights[coalition_size] * marginal_contribution
    return shap_values


def compute_exact_shapley_taylor_values(
    coalition_values: np.ndarray,
    player_count: int,
    order: int,
) -> dict[frozenset[int], np.ndarray]:
    """Exact Shapley-Taylor interaction indices of the given order."""

    if player_count < 1:
        raise ValueError("player_count must be at least 1.")
    if player_count > 20:
        raise ValueError(
            f"Exact Shapley-Taylor requires player_count <= 20; got {player_count}."
        )
    if order < 1 or order > player_count:
        raise ValueError("order must satisfy 1 <= order <= player_count.")

    expected_coalition_count = 1 << player_count
    coalition_array = np.asarray(coalition_values, dtype=float)
    if coalition_array.shape[0] != expected_coalition_count:
        raise ValueError(
            "coalition_values must have one entry per coalition. "
            f"Expected {expected_coalition_count}, got {coalition_array.shape[0]}."
        )
    if np.any(coalition_array[0] != 0):
        warnings.warn(
            "coalition_values[0] is not zero; Shapley-Taylor efficiency at the "
            "empty coalition will not hold exactly.",
            stacklevel=2,
        )

    output_shape = coalition_array.shape[1:]
    all_subset_masks = [
        subset_mask
        for subset_mask in range(1, expected_coalition_count)
        if subset_mask.bit_count() <= order
    ]
    full_mask = expected_coalition_count - 1

    def _discrete_derivative(S_mask: int, T_mask: int) -> np.ndarray:
        out = np.zeros(output_shape, dtype=float)
        W = S_mask
        while True:
            sign = 1 if ((S_mask.bit_count() - W.bit_count()) % 2 == 0) else -1
            out += sign * coalition_array[T_mask | W]
            if W == 0:
                break
            W = (W - 1) & S_mask
        return out

    indices: dict[frozenset[int], np.ndarray] = {}
    for subset_mask in all_subset_masks:
        subset = frozenset(
            player_idx
            for player_idx in range(player_count)
            if subset_mask & (1 << player_idx)
        )
        if subset_mask.bit_count() < order:
            indices[subset] = _discrete_derivative(subset_mask, 0)
            continue

        complement_mask = full_mask ^ subset_mask
        total = np.zeros(output_shape, dtype=float)
        T_mask = complement_mask
        while True:
            subset_size = T_mask.bit_count()
            weight = 1.0 / math.comb(player_count - 1, subset_size)
            total += weight * _discrete_derivative(subset_mask, T_mask)
            if T_mask == 0:
                break
            T_mask = (T_mask - 1) & complement_mask
        indices[subset] = (order / player_count) * total
    return indices


def compute_mobius_transform(
    coalition_values: np.ndarray,
    player_count: int,
) -> np.ndarray:
    expected_coalition_count = 1 << player_count
    mobius = np.asarray(coalition_values, dtype=float).copy()

    if mobius.shape[0] != expected_coalition_count:
        raise ValueError(
            "coalition_values must have one entry per coalition. "
            f"Expected {expected_coalition_count}, got {mobius.shape[0]}."
        )

    for bit in range(player_count):
        bit_mask = 1 << bit
        for mask in range(expected_coalition_count):
            if mask & bit_mask:
                mobius[mask] -= mobius[mask ^ bit_mask]

    return mobius


def compute_exact_faith_shap_values(
    coalition_values: np.ndarray,
    player_count: int,
    order: int,
) -> dict[frozenset[int], np.ndarray]:
    """Exact Faith-SHAP interaction indices up to maximum interaction order."""

    if player_count < 1:
        raise ValueError("player_count must be at least 1.")
    if player_count > 20:
        raise ValueError(
            f"Exact Faith-SHAP requires player_count <= 20; got {player_count}."
        )
    if order < 1 or order > player_count:
        raise ValueError("order must satisfy 1 <= order <= player_count.")

    expected_coalition_count = 1 << player_count
    coalition_array = np.asarray(coalition_values, dtype=float)
    if coalition_array.shape[0] != expected_coalition_count:
        raise ValueError(
            "coalition_values must have one entry per coalition. "
            f"Expected {expected_coalition_count}, got {coalition_array.shape[0]}."
        )

    mobius = compute_mobius_transform(coalition_array, player_count)
    full_mask = expected_coalition_count - 1
    indices: dict[frozenset[int], np.ndarray] = {}

    for subset_mask in range(1, expected_coalition_count):
        subset_size = subset_mask.bit_count()
        if subset_size > order:
            continue

        value = np.asarray(mobius[subset_mask], dtype=float).copy()
        base_coeff = (
            ((-1.0) ** (order - subset_size))
            * (subset_size / (order + subset_size))
            * math.comb(order, subset_size)
        )
        complement_mask = full_mask ^ subset_mask
        extra_mask = complement_mask
        correction = np.zeros_like(value, dtype=float)

        while True:
            superset_mask = subset_mask | extra_mask
            superset_size = superset_mask.bit_count()

            if superset_size > order:
                weight = (
                    math.comb(superset_size - 1, order)
                    / math.comb(superset_size + order - 1, order + subset_size)
                )
                correction += weight * mobius[superset_mask]

            if extra_mask == 0:
                break
            extra_mask = (extra_mask - 1) & complement_mask

        value += base_coeff * correction
        subset = frozenset(
            player_idx
            for player_idx in range(player_count)
            if subset_mask & (1 << player_idx)
        )
        indices[subset] = value

    return indices


def compute_exact_interaction_values(
    coalition_values: np.ndarray,
    player_count: int,
    order: int,
    method: str,
) -> dict[frozenset[int], np.ndarray]:
    if method == "shapley_taylor":
        return compute_exact_shapley_taylor_values(
            coalition_values,
            player_count=player_count,
            order=order,
        )
    if method == "faith_shap":
        return compute_exact_faith_shap_values(
            coalition_values,
            player_count=player_count,
            order=order,
        )
    raise ValueError(
        "method must be one of: " + ", ".join(SUPPORTED_INTERACTION_METHODS)
    )


def _solve_all_coalitions_for_day(
    *,
    observation: pd.Series,
    true_prices: Sequence[float],
    date: str,
    evaluator: Any,
    actual_parameters: StorageDispatchParameters,
    parameter_player_spec: ParameterPlayerSpec | None,
    solver_params: dict[str, Any],
    objective_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, tuple[StorageDispatchResult, ...], StorageDispatchResult | None]:
    """Solve every coalition dispatch for one day.

    When parameter_player_spec is None, this preserves the legacy feature-only
    path. When parameter players are enabled, feature predictions are cached and
    reused across all parameter masks.
    """

    if parameter_player_spec is None:
        coalition_predictions = evaluator.evaluate_all_coalitions(observation)
        coalition_realized_values = np.zeros(coalition_predictions.shape[0], dtype=float)
        coalition_dispatch_results: list[StorageDispatchResult] = []
        full_dispatch_result = None
        for coalition_mask, predicted_prices in enumerate(coalition_predictions):
            dispatch_result = solve_storage_dispatch_lexicographic(
                predicted_prices,
                actual_parameters,
                name=f"storage_dispatch_{date}_{coalition_mask}",
                log_to_console=False,
                solver_params=solver_params,
                objective_tolerance=objective_tolerance,
            )
            realized_evaluation = evaluate_storage_dispatch_result(
                true_prices,
                dispatch_result,
                actual_parameters,
            )
            coalition_realized_values[coalition_mask] = realized_evaluation.objective_value
            coalition_dispatch_results.append(dispatch_result)
            if coalition_mask == coalition_predictions.shape[0] - 1:
                full_dispatch_result = dispatch_result
        return (
            coalition_predictions,
            coalition_realized_values,
            tuple(coalition_dispatch_results),
            full_dispatch_result,
        )

    if not isinstance(evaluator, ExtendedPlayerCoalitionEvaluator):
        raise TypeError(
            "Expected an ExtendedPlayerCoalitionEvaluator when parameter players "
            "are enabled."
        )

    feature_predictions = evaluator.feature_predictions_for_all_feature_coalitions(observation)
    feature_mask_full = (1 << evaluator.feature_count) - 1
    total_coalitions = 1 << evaluator.player_count
    extended_predictions = np.zeros((total_coalitions, feature_predictions.shape[1]), dtype=float)
    coalition_realized_values = np.zeros(total_coalitions, dtype=float)
    coalition_dispatch_results = []
    full_dispatch_result = None

    for coalition_mask in range(total_coalitions):
        feature_mask = coalition_mask & feature_mask_full
        param_mask = coalition_mask >> evaluator.feature_count
        predicted_prices = feature_predictions[feature_mask]
        extended_predictions[coalition_mask] = predicted_prices
        coalition_parameters = evaluator.parameters_for_param_mask(param_mask)
        dispatch_result = solve_storage_dispatch_lexicographic(
            predicted_prices,
            coalition_parameters,
            name=f"storage_dispatch_{date}_{coalition_mask}",
            log_to_console=False,
            solver_params=solver_params,
            objective_tolerance=objective_tolerance,
        )
        # Use each coalition's own parameters for both planning and realized-value
        # evaluation so the game remains self-consistent and avoids replaying a
        # dispatch under incompatible storage constraints.
        realized_evaluation = evaluate_storage_dispatch_result(
            true_prices,
            dispatch_result,
            coalition_parameters,
        )
        coalition_realized_values[coalition_mask] = realized_evaluation.objective_value
        coalition_dispatch_results.append(dispatch_result)
        if coalition_mask == total_coalitions - 1:
            full_dispatch_result = dispatch_result

    return (
        extended_predictions,
        coalition_realized_values,
        tuple(coalition_dispatch_results),
        full_dispatch_result,
    )


def _evaluate_coalition_dispatch_results(
    *,
    valuation_prices: Sequence[float],
    coalition_dispatch_results: Sequence[StorageDispatchResult],
    evaluator: Any,
    actual_parameters: StorageDispatchParameters,
    parameter_player_spec: ParameterPlayerSpec | None,
) -> np.ndarray:
    coalition_values = np.zeros(len(coalition_dispatch_results), dtype=float)
    if parameter_player_spec is None:
        for coalition_mask, dispatch_result in enumerate(coalition_dispatch_results):
            coalition_values[coalition_mask] = evaluate_storage_dispatch_result(
                valuation_prices,
                dispatch_result,
                actual_parameters,
            ).objective_value
        return coalition_values

    if not isinstance(evaluator, ExtendedPlayerCoalitionEvaluator):
        raise TypeError(
            "Expected an ExtendedPlayerCoalitionEvaluator when parameter players "
            "are enabled."
        )

    for coalition_mask, dispatch_result in enumerate(coalition_dispatch_results):
        param_mask = coalition_mask >> evaluator.feature_count
        coalition_parameters = evaluator.parameters_for_param_mask(param_mask)
        coalition_values[coalition_mask] = evaluate_storage_dispatch_result(
            valuation_prices,
            dispatch_result,
            coalition_parameters,
        ).objective_value
    return coalition_values


def _build_decision_evaluation_metric_values(
    *,
    attribution_values: np.ndarray,
    decision_characteristic_values: np.ndarray,
    player_names: Sequence[str],
) -> dict[str, float | None]:
    return {
        "decision_deletion_auc": compute_decision_deletion_auc(
            attribution_values,
            decision_characteristic_values,
            player_names,
        ),
        "decision_insertion_auc": compute_decision_insertion_auc(
            attribution_values,
            decision_characteristic_values,
            player_names,
        ),
        "decision_infidelity": compute_exact_decision_infidelity(
            attribution_values,
            decision_characteristic_values,
            player_names,
        ),
    }


def _validate_caiso_shap_config(
    config: CaisoShapCaseStudyConfig,
    *,
    max_days: int | None,
    holdout_mean_impute_features: Sequence[str],
) -> None:
    if max_days is not None and max_days <= 0:
        raise ValueError("max_days must be strictly positive when provided.")
    if config.background_days <= 0:
        raise ValueError("background_days must be strictly positive.")
    if len(set(holdout_mean_impute_features)) != len(holdout_mean_impute_features):
        raise ValueError("holdout_mean_impute_features must not contain duplicates.")
    if config.model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            "model_name must be one of: " + ", ".join(SUPPORTED_MODEL_NAMES)
        )
    if config.interaction_method not in SUPPORTED_INTERACTION_METHODS:
        raise ValueError(
            "interaction_method must be one of: "
            + ", ".join(SUPPORTED_INTERACTION_METHODS)
        )


def run_caiso_shap_case_study(
    config: CaisoShapCaseStudyConfig,
) -> CaisoShapCaseStudyOutputs:
    _validate_caiso_shap_config(
        config,
        max_days=config.max_days,
        holdout_mean_impute_features=config.holdout_mean_impute_features,
    )
    split = load_default_train_explain_split(
        dataset_path=config.dataset_path,
        holdout_days=config.holdout_days,
    )
    explain_frame = split.explain_frame
    if config.max_days is not None:
        explain_frame = explain_frame.iloc[: config.max_days].reset_index(drop=True)
    if explain_frame.empty:
        raise ValueError("No explanation rows remain after applying max_days.")
    background_frame = select_recent_background_frame(
        split.train_frame,
        split.date_column,
        config.background_days,
    )
    training_artifacts = train_model(
        split.X_train,
        split.y_train,
        model_name=config.model_name,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        mlp_hidden_layer_sizes=config.mlp_hidden_layer_sizes,
        mlp_max_iter=config.mlp_max_iter,
        mlp_dropout=config.mlp_dropout,
        mlp_weight_decay=config.mlp_weight_decay,
        mlp_batch_size=config.mlp_batch_size,
        mlp_early_stopping_patience=config.mlp_early_stopping_patience,
        mlp_activation=config.mlp_activation,
        mlp_batch_norm=config.mlp_batch_norm,
        xgb_n_estimators=config.xgb_n_estimators,
        xgb_max_depth=config.xgb_max_depth,
        xgb_learning_rate=config.xgb_learning_rate,
        xgb_subsample=config.xgb_subsample,
        xgb_colsample_bytree=config.xgb_colsample_bytree,
        xgb_reg_lambda=config.xgb_reg_lambda,
        xgb_verbosity=config.xgb_verbosity,
        learning_rate=config.learning_rate,
        mse_learning_rate=config.mse_learning_rate,
        spo_learning_rate=config.spo_learning_rate,
        storage_parameters=config.storage_parameters,
        training_verbose=config.training_verbose,
        training_log_every=config.training_log_every,
        spo_processes=config.spo_processes,
        spo_warm_start_with_mse=config.spo_warm_start_with_mse,
    )
    return run_caiso_shap_case_study_with_artifacts(
        config=config,
        training_artifacts=training_artifacts,
        dataset_path=split.dataset_path,
        date_column=split.date_column,
        feature_columns=split.feature_columns,
        target_columns=split.target_columns,
        train_frame=split.train_frame,
        background_frame=background_frame,
        explain_frame=explain_frame,
        holdout_mean_impute_features=config.holdout_mean_impute_features,
        holdout_days=config.holdout_days,
        max_days=config.max_days,
    )


def run_caiso_shap_case_study_with_artifacts(
    *,
    config: CaisoShapCaseStudyConfig,
    training_artifacts: ModelTrainingArtifacts,
    dataset_path: Path | str,
    date_column: str,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    train_frame: pd.DataFrame,
    background_frame: pd.DataFrame,
    explain_frame: pd.DataFrame,
    holdout_mean_impute_features: Sequence[str] = (),
    holdout_days: int | None = None,
    max_days: int | None = None,
    evaluation_label: str | None = None,
) -> CaisoShapCaseStudyOutputs:
    feature_columns = tuple(feature_columns)
    target_columns = tuple(target_columns)
    holdout_mean_impute_features = tuple(holdout_mean_impute_features)
    _validate_caiso_shap_config(
        config,
        max_days=max_days,
        holdout_mean_impute_features=holdout_mean_impute_features,
    )
    if train_frame.empty:
        raise ValueError("train_frame must contain at least one row.")
    if background_frame.empty:
        raise ValueError("background_frame must contain at least one row.")
    if explain_frame.empty:
        raise ValueError("No explanation rows remain after applying max_days.")
    if tuple(training_artifacts.feature_columns) != feature_columns:
        raise ValueError("training_artifacts.feature_columns must match feature_columns.")
    if tuple(training_artifacts.target_columns) != target_columns:
        raise ValueError("training_artifacts.target_columns must match target_columns.")

    X_background = background_frame.loc[:, list(feature_columns)]
    unknown_impute_features = sorted(
        set(holdout_mean_impute_features) - set(feature_columns)
    )
    if unknown_impute_features:
        raise ValueError(
            "holdout_mean_impute_features contains unknown feature(s): "
            + ", ".join(unknown_impute_features)
        )
    explain_dates = explain_frame.loc[:, date_column].tolist()
    X_explain = explain_frame.loc[:, list(feature_columns)]
    y_explain = explain_frame.loc[:, list(target_columns)]
    holdout_feature_replacements: dict[str, float] = {}
    if holdout_mean_impute_features:
        X_explain = X_explain.copy().astype(float)
        for feature_name in holdout_mean_impute_features:
            replacement_value = float(X_background.loc[:, feature_name].mean())
            X_explain.loc[:, feature_name] = replacement_value
            holdout_feature_replacements[feature_name] = replacement_value
    resolved_player_count = len(feature_columns) + (
        config.parameter_player_spec.player_count
        if config.parameter_player_spec is not None
        else 0
    )
    if config.interaction_order is not None:
        if config.interaction_order < 1:
            raise ValueError("interaction_order must be >= 1 when provided.")
        if config.interaction_order > resolved_player_count:
            raise ValueError(
                f"interaction_order ({config.interaction_order}) exceeds "
                f"total player count ({resolved_player_count})."
            )
        if resolved_player_count > 20:
            raise ValueError(
                f"Exact interaction indices require n_players <= 20; got {resolved_player_count}."
            )

    coalition_evaluator, coalition_expectation_method = _build_coalition_evaluator(
        training_artifacts,
        config,
        X_background,
    )
    is_extended = isinstance(coalition_evaluator, ExtendedPlayerCoalitionEvaluator)
    player_names = (
        coalition_evaluator.player_names()
        if is_extended
        else tuple(feature_columns)
    )
    player_count = len(player_names)
    y_pred_explain = np.asarray(
        training_artifacts.model.predict(X_explain),
        dtype=float,
    )
    if y_pred_explain.ndim == 1:
        y_pred_explain = y_pred_explain[:, np.newaxis]
    y_true_explain = y_explain.to_numpy(dtype=float, copy=True)
    solver_params = {
        "Threads": 1,
        "Seed": config.solver_seed,
        "MIPGap": config.mip_gap,
        "MIPGapAbs": config.mip_gap_abs,
    }
    started_at = time.perf_counter()
    daily_explanations: list[DailyShapExplanation] = []
    daily_interactions: list[DailyInteractionExplanation] = []
    actual_daily_regrets: list[float] = []
    predictive_hourly_rows: list[dict[str, Any]] = []
    daily_full_dispatch_rows: list[dict[str, Any]] = []

    for row_idx, date in enumerate(explain_dates):
        observation = X_explain.iloc[row_idx]
        true_prices = tuple(
            float(value)
            for value in y_explain.iloc[row_idx].to_numpy(dtype=float, copy=True)
        )
        (
            coalition_predictions,
            coalition_realized_values,
            coalition_dispatch_results,
            full_dispatch_result,
        ) = _solve_all_coalitions_for_day(
            observation=observation,
            true_prices=true_prices,
            date=date,
            evaluator=coalition_evaluator,
            actual_parameters=config.storage_parameters,
            parameter_player_spec=config.parameter_player_spec,
            solver_params=solver_params,
            objective_tolerance=config.objective_tolerance,
        )
        predictive_hourly_shap = compute_exact_shapley_values(
            coalition_predictions,
            feature_count=player_count,
        )
        predictive_daily_shap = predictive_hourly_shap.sum(axis=1)

        decision_characteristic_values = (
            coalition_realized_values - coalition_realized_values[0]
        )
        decision_shap = compute_exact_shapley_values(
            decision_characteristic_values,
            feature_count=player_count,
        )
        ead_decision_values: np.ndarray | None = None
        ead_decision_characteristic_values: np.ndarray | None = None
        ead_decision_shap: np.ndarray | None = None
        ead_decision_baseline_value: float | None = None
        ead_decision_full_value: float | None = None
        ead_decision_value_gain: float | None = None
        if config.compute_ead_decision_shap:
            ead_decision_values = _evaluate_coalition_dispatch_results(
                valuation_prices=coalition_predictions[-1],
                coalition_dispatch_results=coalition_dispatch_results,
                evaluator=coalition_evaluator,
                actual_parameters=config.storage_parameters,
                parameter_player_spec=config.parameter_player_spec,
            )
            ead_decision_characteristic_values = (
                ead_decision_values - ead_decision_values[0]
            )
            ead_decision_shap = compute_exact_shapley_values(
                ead_decision_characteristic_values,
                feature_count=player_count,
            )
            ead_decision_baseline_value = float(ead_decision_values[0])
            ead_decision_full_value = float(ead_decision_values[-1])
            ead_decision_value_gain = float(ead_decision_characteristic_values[-1])
        decision_activation = compute_decision_activation_metrics(
            decision_characteristic_values,
            coalition_dispatch_results,
            player_count,
            _storage_dispatch_decision_changed,
        )
        evaluation_metrics_by_family = {
            "predictive": _build_decision_evaluation_metric_values(
                attribution_values=predictive_daily_shap,
                decision_characteristic_values=decision_characteristic_values,
                player_names=player_names,
            ),
            "decision": _build_decision_evaluation_metric_values(
                attribution_values=decision_shap,
                decision_characteristic_values=decision_characteristic_values,
                player_names=player_names,
            ),
        }
        if (
            ead_decision_shap is not None
            and ead_decision_characteristic_values is not None
        ):
            evaluation_metrics_by_family["ead_decision"] = (
                _build_decision_evaluation_metric_values(
                    attribution_values=ead_decision_shap,
                    decision_characteristic_values=ead_decision_characteristic_values,
                    player_names=player_names,
                )
            )

        predictive_ranking = build_attribution_ranking(
            predictive_daily_shap,
            player_names,
        )
        decision_ranking = build_attribution_ranking(
            decision_shap,
            player_names,
        )
        daily_abs_rank_spearman = compute_rank_spearman_from_rankings(
            predictive_ranking,
            decision_ranking,
        )
        daily_abs_rank_kendall_tau = compute_rank_kendall_tau_from_rankings(
            predictive_ranking,
            decision_ranking,
        )
        predictive_ead_abs_rank_spearman = None
        predictive_ead_abs_rank_kendall_tau = None
        decision_ead_abs_rank_spearman = None
        decision_ead_abs_rank_kendall_tau = None
        if ead_decision_shap is not None:
            ead_decision_ranking = build_attribution_ranking(
                ead_decision_shap,
                player_names,
            )
            predictive_ead_abs_rank_spearman = compute_rank_spearman_from_rankings(
                predictive_ranking,
                ead_decision_ranking,
            )
            predictive_ead_abs_rank_kendall_tau = (
                compute_rank_kendall_tau_from_rankings(
                    predictive_ranking,
                    ead_decision_ranking,
                )
            )
            decision_ead_abs_rank_spearman = compute_rank_spearman_from_rankings(
                decision_ranking,
                ead_decision_ranking,
            )
            decision_ead_abs_rank_kendall_tau = (
                compute_rank_kendall_tau_from_rankings(
                    decision_ranking,
                    ead_decision_ranking,
                )
            )
        if full_dispatch_result is None:
            raise RuntimeError("Expected to capture the full-coalition dispatch result.")

        if config.interaction_order is not None:
            decision_raw = compute_exact_interaction_values(
                decision_characteristic_values,
                player_count=player_count,
                order=config.interaction_order,
                method=config.interaction_method,
            )
            predictive_raw = compute_exact_interaction_values(
                coalition_predictions,
                player_count=player_count,
                order=config.interaction_order,
                method=config.interaction_method,
            )
            daily_interactions.append(
                DailyInteractionExplanation(
                    date=date,
                    method=config.interaction_method,
                    order=config.interaction_order,
                    player_names=player_names,
                    decision_indices={
                        subset: float(np.asarray(value, dtype=float))
                        for subset, value in decision_raw.items()
                    },
                    predictive_indices={
                        subset: np.asarray(value, dtype=float)
                        for subset, value in predictive_raw.items()
                    },
                    decision_value_full=float(decision_characteristic_values[-1]),
                    predictive_value_full=np.asarray(coalition_predictions[-1], dtype=float),
                    predictive_value_empty=np.asarray(coalition_predictions[0], dtype=float),
                )
            )

        oracle_dispatch_result = solve_storage_dispatch_lexicographic(
            true_prices,
            config.storage_parameters,
            name=f"storage_dispatch_oracle_{date}",
            log_to_console=False,
            solver_params=solver_params,
            objective_tolerance=config.objective_tolerance,
        )
        oracle_objective_value = float(oracle_dispatch_result.objective_value)
        decision_full_value = float(coalition_realized_values[-1])
        daily_explanations.append(
            DailyShapExplanation(
                date=date,
                predictive_hourly_shap=predictive_hourly_shap,
                predictive_daily_shap=predictive_daily_shap,
                decision_shap=decision_shap,
                ead_decision_shap=ead_decision_shap,
                decision_activation_rate=decision_activation.activation_rate,
                decision_activated_value_sum=decision_activation.activated_value_sum,
                decision_activated_value=decision_activation.activated_value,
                evaluation_metrics_by_family=evaluation_metrics_by_family,
                predictive_baseline_total=float(coalition_predictions[0].sum()),
                predictive_full_total=float(coalition_predictions[-1].sum()),
                decision_baseline_value=float(coalition_realized_values[0]),
                decision_full_value=decision_full_value,
                ead_decision_baseline_value=ead_decision_baseline_value,
                ead_decision_full_value=ead_decision_full_value,
                ead_decision_characteristic_baseline_value=(
                    0.0 if ead_decision_shap is not None else None
                ),
                ead_decision_characteristic_full_value=ead_decision_value_gain,
                ead_decision_value_gain=ead_decision_value_gain,
                oracle_objective_value=oracle_objective_value,
                decision_value_gain=float(
                    coalition_realized_values[-1] - coalition_realized_values[0]
                ),
                daily_abs_rank_spearman=daily_abs_rank_spearman,
                daily_abs_rank_kendall_tau=daily_abs_rank_kendall_tau,
                predictive_ead_abs_rank_spearman=predictive_ead_abs_rank_spearman,
                predictive_ead_abs_rank_kendall_tau=predictive_ead_abs_rank_kendall_tau,
                decision_ead_abs_rank_spearman=decision_ead_abs_rank_spearman,
                decision_ead_abs_rank_kendall_tau=decision_ead_abs_rank_kendall_tau,
            )
        )
        actual_daily_regrets.append(
            oracle_objective_value - decision_full_value
        )

        for feature_idx, feature_name in enumerate(player_names):
            for hour_idx, shap_value in enumerate(
                predictive_hourly_shap[feature_idx],
                start=1,
            ):
                predictive_hourly_rows.append(
                    {
                        "date": date,
                        "feature": feature_name,
                        "hour": hour_idx,
                        "target_column": target_columns[hour_idx - 1],
                        "shap_value": float(shap_value),
                    }
                )

        daily_full_dispatch_rows.extend(
            _build_daily_full_dispatch_rows(date, full_dispatch_result)
        )

    total_runtime_seconds = time.perf_counter() - started_at
    daily_shap = _build_daily_shap_frame(daily_explanations, player_names)
    predictive_hourly_shap = pd.DataFrame(predictive_hourly_rows)
    daily_full_dispatch = pd.DataFrame(daily_full_dispatch_rows)
    summary_shap = _build_summary_shap_frame(daily_shap, player_names)
    prediction_metrics = _build_prediction_metrics(
        y_true=y_true_explain,
        y_pred=y_pred_explain,
        decision_value_gains=[
            explanation.decision_value_gain for explanation in daily_explanations
        ],
        actual_daily_regrets=actual_daily_regrets,
    )
    comparison_metrics = _build_comparison_metrics(
        daily_explanations=daily_explanations,
        summary_shap=summary_shap,
    )
    evaluation_metrics = _build_evaluation_metrics(
        daily_explanations,
    )
    daily_interaction_decision = _build_daily_interaction_decision_frame(
        daily_interactions
    )
    daily_interaction_predictive = _build_daily_interaction_predictive_frame(
        daily_interactions
    )
    daily_shapley_taylor_decision = (
        _build_legacy_shapley_taylor_decision_frame(daily_interaction_decision)
        if config.interaction_method == "shapley_taylor"
        else None
    )
    daily_shapley_taylor_predictive = (
        _build_legacy_shapley_taylor_predictive_frame(daily_interaction_predictive)
        if config.interaction_method == "shapley_taylor"
        else None
    )
    interaction_efficiency_gap_decision = (
        max(
            abs(
                sum(explanation.decision_indices.values())
                - explanation.decision_value_full
            )
            for explanation in daily_interactions
        )
        if daily_interactions
        else None
    )
    interaction_efficiency_gap_predictive = (
        max(
            float(
                np.max(
                    np.abs(
                        sum(explanation.predictive_indices.values())
                        - (
                            explanation.predictive_value_full
                            - explanation.predictive_value_empty
                        )
                    )
                )
            )
            for explanation in daily_interactions
        )
        if daily_interactions
        else None
    )
    run_metadata = {
        "dataset_path": str(dataset_path),
        "evaluation_label": evaluation_label,
        "model_name": training_artifacts.model_name,
        "model_description": training_artifacts.model_description,
        "coalition_expectation_method": coalition_expectation_method,
        "feature_columns": list(feature_columns),
        "player_count": player_count,
        "player_names": list(player_names),
        "target_columns": list(target_columns),
        "train_date_start": train_frame.loc[:, date_column].iloc[0],
        "train_date_end": train_frame.loc[:, date_column].iloc[-1],
        "background_date_start": str(background_frame[date_column].iloc[0]),
        "background_date_end": str(background_frame[date_column].iloc[-1]),
        "explain_date_start": explain_dates[0],
        "explain_date_end": explain_dates[-1],
        "train_rows": int(len(train_frame)),
        "background_rows": int(len(X_background)),
        "explain_rows": int(len(X_explain)),
        "holdout_days": config.holdout_days if holdout_days is None else holdout_days,
        "background_days": config.background_days,
        "holdout_mean_impute_features": list(holdout_mean_impute_features),
        "holdout_feature_replacements": holdout_feature_replacements,
        "holdout_feature_replacement_strategy": (
            "background_mean" if holdout_feature_replacements else None
        ),
        "max_days": max_days,
        "coalitions_per_day": coalition_evaluator.coalition_count,
        "compute_ead_decision_shap": bool(config.compute_ead_decision_shap),
        "ead_decision_shap_family": (
            "positive_full_prediction_ex_ante_decision_value"
            if config.compute_ead_decision_shap
            else None
        ),
        "ead_decision_characteristic_function": (
            "v_ante(S)=J(yhat_N,w(S))-J(yhat_N,w(empty))"
            if config.compute_ead_decision_shap
            else None
        ),
        "interaction_order": config.interaction_order,
        "interaction_method": (
            None if config.interaction_order is None else config.interaction_method
        ),
        "parameter_player_spec": (
            None
            if config.parameter_player_spec is None
            else dataclasses.asdict(config.parameter_player_spec)
        ),
        "storage_parameters": {
            "energy_capacity": config.storage_parameters.energy_capacity,
            "power_limit": config.storage_parameters.power_limit,
            "charge_efficiency": config.storage_parameters.charge_efficiency,
            "discharge_efficiency": config.storage_parameters.discharge_efficiency,
            "throughput_penalty": config.storage_parameters.throughput_penalty,
            "initial_state_of_charge": config.storage_parameters.initial_state_of_charge,
            "terminal_state_of_charge": config.storage_parameters.terminal_state_of_charge,
        },
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
        "mlp_hidden_layer_sizes": list(config.mlp_hidden_layer_sizes),
        "mlp_max_iter": config.mlp_max_iter,
        "mlp_dropout": config.mlp_dropout,
        "mlp_weight_decay": config.mlp_weight_decay,
        "mlp_batch_size": config.mlp_batch_size,
        "mlp_early_stopping_patience": config.mlp_early_stopping_patience,
        "mlp_activation": config.mlp_activation,
        "mlp_batch_norm": config.mlp_batch_norm,
        "xgb_params": _build_caiso_xgb_params(config),
        "learning_rate": config.learning_rate,
        "mse_learning_rate": config.mse_learning_rate,
        "spo_learning_rate": config.spo_learning_rate,
        "training_verbose": config.training_verbose,
        "training_log_every": config.training_log_every,
        "spo_processes": config.spo_processes,
        "spo_warm_start_with_mse": config.spo_warm_start_with_mse,
        "solver_params": solver_params,
        "objective_tolerance": config.objective_tolerance,
        "decision_activation_metric_enabled": True,
        "decision_activation_definition": "charge/discharge dispatch change",
        "decision_activation_dispatch_atol": CAISO_DECISION_ACTIVATION_DISPATCH_ATOL,
        "evaluation_metric_parameters": {
            "decision_deletion_auc_clipping": DECISION_DELETION_AUC_CLIPPING,
            "decision_deletion_auc_lower_is_better": True,
            "decision_deletion_auc_requires_positive_full_gain": True,
            "decision_deletion_auc_zero_if_all_strict_suffixes_nonpositive": True,
            "decision_insertion_auc_clipping": DECISION_INSERTION_AUC_CLIPPING,
            "decision_insertion_auc_requires_positive_full_gain": True,
            "decision_insertion_auc_zero_if_all_strict_prefixes_nonpositive": True,
        },
        "interaction_efficiency_gap_decision": (
            None
            if interaction_efficiency_gap_decision is None
            else float(interaction_efficiency_gap_decision)
        ),
        "interaction_efficiency_gap_predictive": (
            None
            if interaction_efficiency_gap_predictive is None
            else float(interaction_efficiency_gap_predictive)
        ),
        "shapley_taylor_efficiency_gap_decision": (
            None
            if config.interaction_method != "shapley_taylor"
            else (
                None
                if interaction_efficiency_gap_decision is None
                else float(interaction_efficiency_gap_decision)
            )
        ),
        "runtime_seconds": total_runtime_seconds,
    }

    return CaisoShapCaseStudyOutputs(
        daily_shap=daily_shap,
        predictive_hourly_shap=predictive_hourly_shap,
        daily_full_dispatch=daily_full_dispatch,
        summary_shap=summary_shap,
        prediction_metrics=prediction_metrics,
        comparison_metrics=comparison_metrics,
        evaluation_metrics=evaluation_metrics,
        run_metadata=run_metadata,
        daily_interaction_decision=daily_interaction_decision,
        daily_interaction_predictive=daily_interaction_predictive,
        daily_shapley_taylor_decision=daily_shapley_taylor_decision,
        daily_shapley_taylor_predictive=daily_shapley_taylor_predictive,
    )


def write_caiso_shap_case_study_outputs(
    outputs: CaisoShapCaseStudyOutputs,
    outdir: Path | str,
) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    outputs.daily_shap.to_csv(outdir_path / "daily_shap.csv", index=False)
    outputs.predictive_hourly_shap.to_csv(
        outdir_path / "predictive_hourly_shap.csv",
        index=False,
    )
    outputs.daily_full_dispatch.to_csv(
        outdir_path / "daily_full_dispatch.csv",
        index=False,
    )
    outputs.summary_shap.to_csv(outdir_path / "summary_shap.csv", index=False)

    with (outdir_path / "prediction_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.prediction_metrics, handle, indent=2, sort_keys=True)

    with (outdir_path / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.comparison_metrics, handle, indent=2, sort_keys=True)

    with (outdir_path / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.evaluation_metrics, handle, indent=2, sort_keys=True)

    with (outdir_path / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.run_metadata, handle, indent=2, sort_keys=True)

    if outputs.daily_interaction_decision is not None:
        outputs.daily_interaction_decision.to_csv(
            outdir_path / "daily_interaction_decision.csv",
            index=False,
        )
    if outputs.daily_interaction_predictive is not None:
        outputs.daily_interaction_predictive.to_csv(
            outdir_path / "daily_interaction_predictive.csv",
            index=False,
        )

    if outputs.daily_shapley_taylor_decision is not None:
        outputs.daily_shapley_taylor_decision.to_csv(
            outdir_path / "daily_shapley_taylor_decision.csv",
            index=False,
        )
    if outputs.daily_shapley_taylor_predictive is not None:
        outputs.daily_shapley_taylor_predictive.to_csv(
            outdir_path / "daily_shapley_taylor_predictive.csv",
            index=False,
        )
    if (
        outputs.run_metadata.get("interaction_method") == "faith_shap"
        and outputs.daily_interaction_decision is not None
    ):
        outputs.daily_interaction_decision.to_csv(
            outdir_path / "daily_faith_shap_decision.csv",
            index=False,
        )
    if (
        outputs.run_metadata.get("interaction_method") == "faith_shap"
        and outputs.daily_interaction_predictive is not None
    ):
        outputs.daily_interaction_predictive.to_csv(
            outdir_path / "daily_faith_shap_predictive.csv",
            index=False,
        )


def _build_daily_shap_frame(
    daily_explanations: Sequence[DailyShapExplanation],
    player_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for explanation in daily_explanations:
        row: dict[str, Any] = {
            "date": explanation.date,
            "predictive_baseline_total": explanation.predictive_baseline_total,
            "predictive_full_total": explanation.predictive_full_total,
            "predictive_total_gain": (
                explanation.predictive_full_total - explanation.predictive_baseline_total
            ),
            "decision_baseline_value": explanation.decision_baseline_value,
            "decision_full_value": explanation.decision_full_value,
            "oracle_obj": explanation.oracle_objective_value,
            "decision_value_gain": explanation.decision_value_gain,
            "abs_rank_spearman": explanation.daily_abs_rank_spearman,
            "abs_rank_kendall_tau": explanation.daily_abs_rank_kendall_tau,
        }
        if explanation.ead_decision_shap is not None:
            row.update(
                {
                    "ead_decision_baseline_value": (
                        explanation.ead_decision_baseline_value
                    ),
                    "ead_decision_full_value": explanation.ead_decision_full_value,
                    "ead_decision_characteristic_baseline_value": (
                        explanation.ead_decision_characteristic_baseline_value
                    ),
                    "ead_decision_characteristic_full_value": (
                        explanation.ead_decision_characteristic_full_value
                    ),
                    "ead_decision_value_gain": explanation.ead_decision_value_gain,
                    "predictive_ead_abs_rank_spearman": (
                        explanation.predictive_ead_abs_rank_spearman
                    ),
                    "predictive_ead_abs_rank_kendall_tau": (
                        explanation.predictive_ead_abs_rank_kendall_tau
                    ),
                    "decision_ead_abs_rank_spearman": (
                        explanation.decision_ead_abs_rank_spearman
                    ),
                    "decision_ead_abs_rank_kendall_tau": (
                        explanation.decision_ead_abs_rank_kendall_tau
                    ),
                }
            )
        for feature_name, shap_value in zip(
            player_names,
            explanation.predictive_daily_shap,
            strict=True,
        ):
            row[f"predictive_shap_{feature_name}"] = float(shap_value)
        for feature_name, shap_value in zip(
            player_names,
            explanation.decision_shap,
            strict=True,
        ):
            row[f"decision_shap_{feature_name}"] = float(shap_value)
        if explanation.ead_decision_shap is not None:
            for feature_name, shap_value in zip(
                player_names,
                explanation.ead_decision_shap,
                strict=True,
            ):
                row[f"ead_decision_shap_{feature_name}"] = float(shap_value)
        for feature_name, activation_rate, activated_value_sum, activated_value in zip(
            player_names,
            explanation.decision_activation_rate,
            explanation.decision_activated_value_sum,
            explanation.decision_activated_value,
            strict=True,
        ):
            row[f"decision_activation_rate_{feature_name}"] = float(activation_rate)
            row[f"decision_activated_value_sum_{feature_name}"] = float(
                activated_value_sum
            )
            row[f"decision_activated_value_{feature_name}"] = float(activated_value)
        for explainer_family, metric_values in explanation.evaluation_metrics_by_family.items():
            for metric_name, metric_value in metric_values.items():
                row[f"{explainer_family}_{metric_name}"] = (
                    None if metric_value is None else float(metric_value)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_daily_interaction_decision_frame(
    daily_explanations: Sequence[DailyInteractionExplanation],
) -> pd.DataFrame | None:
    if not daily_explanations:
        return None

    rows: list[dict[str, Any]] = []
    for explanation in daily_explanations:
        for subset, value in explanation.decision_indices.items():
            rows.append(
                {
                    "date": explanation.date,
                    "interaction_method": explanation.method,
                    "order": explanation.order,
                    "subset_size": len(subset),
                    "players": _format_player_subset(subset, explanation.player_names),
                    "decision_interaction_value": float(value),
                }
            )
    return pd.DataFrame(rows)


def _build_daily_interaction_predictive_frame(
    daily_explanations: Sequence[DailyInteractionExplanation],
) -> pd.DataFrame | None:
    if not daily_explanations:
        return None

    rows: list[dict[str, Any]] = []
    for explanation in daily_explanations:
        for subset, values in explanation.predictive_indices.items():
            player_subset = _format_player_subset(subset, explanation.player_names)
            for hour_idx, value in enumerate(values, start=1):
                rows.append(
                    {
                        "date": explanation.date,
                        "interaction_method": explanation.method,
                        "order": explanation.order,
                        "subset_size": len(subset),
                        "players": player_subset,
                        "hour": hour_idx,
                        "predictive_interaction_value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _build_legacy_shapley_taylor_decision_frame(
    daily_interaction_decision: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if daily_interaction_decision is None:
        return None
    return daily_interaction_decision.loc[
        :,
        ["date", "order", "subset_size", "players", "decision_interaction_value"],
    ].rename(columns={"decision_interaction_value": "decision_shapley_taylor"})


def _build_legacy_shapley_taylor_predictive_frame(
    daily_interaction_predictive: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if daily_interaction_predictive is None:
        return None
    return daily_interaction_predictive.loc[
        :,
        [
            "date",
            "order",
            "subset_size",
            "players",
            "hour",
            "predictive_interaction_value",
        ],
    ].rename(columns={"predictive_interaction_value": "predictive_shapley_taylor"})


def _build_daily_full_dispatch_rows(
    date: str,
    full_dispatch_result: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hour_idx, (charge, discharge, mode) in enumerate(
        zip(
            full_dispatch_result.charge,
            full_dispatch_result.discharge,
            full_dispatch_result.mode,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "date": date,
                "hour": hour_idx,
                "charge": float(charge),
                "discharge": float(discharge),
                "mode": int(mode),
                "state_of_charge_start": float(full_dispatch_result.state_of_charge[hour_idx - 1]),
                "state_of_charge_end": float(full_dispatch_result.state_of_charge[hour_idx]),
            }
        )
    return rows


def _storage_dispatch_decision_changed(
    left_result: StorageDispatchResult,
    right_result: StorageDispatchResult,
) -> bool:
    left_decision = np.concatenate(
        (
            np.asarray(left_result.charge, dtype=float),
            np.asarray(left_result.discharge, dtype=float),
        )
    )
    right_decision = np.concatenate(
        (
            np.asarray(right_result.charge, dtype=float),
            np.asarray(right_result.discharge, dtype=float),
        )
    )
    if left_decision.shape != right_decision.shape:
        raise ValueError("Dispatch decisions must have the same shape.")
    if left_decision.size == 0:
        return False
    return bool(
        np.max(np.abs(left_decision - right_decision))
        > CAISO_DECISION_ACTIVATION_DISPATCH_ATOL
    )


def _build_caiso_xgb_params(config: CaisoShapCaseStudyConfig) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": int(config.xgb_n_estimators),
        "max_depth": int(config.xgb_max_depth),
        "learning_rate": float(config.xgb_learning_rate),
        "subsample": float(config.xgb_subsample),
        "colsample_bytree": float(config.xgb_colsample_bytree),
        "reg_lambda": float(config.xgb_reg_lambda),
        "tree_method": "hist",
        "random_state": config.random_state,
        "n_jobs": 1,
        "verbosity": int(config.xgb_verbosity),
    }


def _build_coalition_evaluator(
    training_artifacts: ModelTrainingArtifacts,
    config: CaisoShapCaseStudyConfig,
    background_data: pd.DataFrame,
) -> tuple[Any, str]:
    # Keep all model families, including RF, on the same empirical background
    # marginalization path for direct method comparisons and reproducibility.
    base_evaluator = BackgroundMarginalCoalitionEvaluator(
        training_artifacts.model,
        training_artifacts.feature_columns,
        background_data,
    )
    if config.parameter_player_spec is None:
        return base_evaluator, "empirical_background_marginalization"
    return (
        ExtendedPlayerCoalitionEvaluator(
            feature_evaluator=base_evaluator,
            feature_names=training_artifacts.feature_columns,
            actual_parameters=config.storage_parameters,
            parameter_player_spec=config.parameter_player_spec,
        ),
        "empirical_background_marginalization_with_parameter_players",
    )


def _build_summary_shap_frame(
    daily_shap: pd.DataFrame,
    player_names: Sequence[str],
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    predictive_abs_means: dict[str, float] = {}
    decision_abs_means: dict[str, float] = {}
    ead_decision_abs_means: dict[str, float] = {}
    has_ead_decision = all(
        f"ead_decision_shap_{feature_name}" in daily_shap.columns
        for feature_name in player_names
    )

    for feature_name in player_names:
        predictive_column = daily_shap[f"predictive_shap_{feature_name}"]
        decision_column = daily_shap[f"decision_shap_{feature_name}"]
        predictive_abs_means[feature_name] = float(predictive_column.abs().mean())
        decision_abs_means[feature_name] = float(decision_column.abs().mean())
        if has_ead_decision:
            ead_decision_column = daily_shap[f"ead_decision_shap_{feature_name}"]
            ead_decision_abs_means[feature_name] = float(
                ead_decision_column.abs().mean()
            )

    predictive_ranks = _descending_rank_map(predictive_abs_means)
    decision_ranks = _descending_rank_map(decision_abs_means)
    ead_decision_ranks = (
        _descending_rank_map(ead_decision_abs_means)
        if has_ead_decision
        else {}
    )

    for feature_name in player_names:
        predictive_column = daily_shap[f"predictive_shap_{feature_name}"]
        decision_column = daily_shap[f"decision_shap_{feature_name}"]
        activation_column = daily_shap[f"decision_activation_rate_{feature_name}"]
        activated_value_sum_column = daily_shap[
            f"decision_activated_value_sum_{feature_name}"
        ]
        activation_rate_sum = float(activation_column.sum())
        decision_activated_value = (
            0.0
            if activation_rate_sum <= 0.0
            else float(activated_value_sum_column.sum() / activation_rate_sum)
        )
        row = {
            "feature": feature_name,
            "predictive_mean_signed_shap": float(predictive_column.mean()),
            "predictive_mean_abs_shap": predictive_abs_means[feature_name],
            "predictive_rank": predictive_ranks[feature_name],
            "decision_mean_signed_shap": float(decision_column.mean()),
            "decision_mean_abs_shap": decision_abs_means[feature_name],
            "decision_rank": decision_ranks[feature_name],
            "decision_activation_rate": float(activation_column.mean()),
            "decision_activated_value": decision_activated_value,
        }
        if has_ead_decision:
            ead_decision_column = daily_shap[f"ead_decision_shap_{feature_name}"]
            row.update(
                {
                    "ead_decision_mean_signed_shap": float(
                        ead_decision_column.mean()
                    ),
                    "ead_decision_mean_abs_shap": ead_decision_abs_means[
                        feature_name
                    ],
                    "ead_decision_rank": ead_decision_ranks[feature_name],
                }
            )
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("predictive_rank").reset_index(drop=True)


def _build_comparison_metrics(
    *,
    daily_explanations: Sequence[DailyShapExplanation],
    summary_shap: pd.DataFrame,
) -> dict[str, Any]:
    daily_spearman_by_date = {
        explanation.date: explanation.daily_abs_rank_spearman
        for explanation in daily_explanations
    }
    daily_kendall_tau_by_date = {
        explanation.date: explanation.daily_abs_rank_kendall_tau
        for explanation in daily_explanations
    }
    global_spearman = compute_spearman_rank_correlation(
        summary_shap["predictive_mean_abs_shap"].to_numpy(dtype=float),
        summary_shap["decision_mean_abs_shap"].to_numpy(dtype=float),
    )
    global_kendall_tau = compute_kendall_tau_correlation(
        summary_shap["predictive_mean_abs_shap"].to_numpy(dtype=float),
        summary_shap["decision_mean_abs_shap"].to_numpy(dtype=float),
    )
    metrics = {
        "daily_abs_rank_spearman": _build_rank_agreement_summary(
            daily_spearman_by_date,
        ),
        "daily_abs_rank_kendall_tau": _build_rank_agreement_summary(
            daily_kendall_tau_by_date,
        ),
        "global_abs_rank_spearman": global_spearman,
        "global_abs_rank_kendall_tau": global_kendall_tau,
    }
    if "ead_decision_mean_abs_shap" in summary_shap.columns:
        predictive_ead_spearman_by_date = {
            explanation.date: explanation.predictive_ead_abs_rank_spearman
            for explanation in daily_explanations
            if explanation.ead_decision_shap is not None
        }
        predictive_ead_kendall_by_date = {
            explanation.date: explanation.predictive_ead_abs_rank_kendall_tau
            for explanation in daily_explanations
            if explanation.ead_decision_shap is not None
        }
        decision_ead_spearman_by_date = {
            explanation.date: explanation.decision_ead_abs_rank_spearman
            for explanation in daily_explanations
            if explanation.ead_decision_shap is not None
        }
        decision_ead_kendall_by_date = {
            explanation.date: explanation.decision_ead_abs_rank_kendall_tau
            for explanation in daily_explanations
            if explanation.ead_decision_shap is not None
        }
        metrics.update(
            {
                "daily_predictive_ead_abs_rank_spearman": (
                    _build_rank_agreement_summary(predictive_ead_spearman_by_date)
                ),
                "daily_predictive_ead_abs_rank_kendall_tau": (
                    _build_rank_agreement_summary(predictive_ead_kendall_by_date)
                ),
                "daily_decision_ead_abs_rank_spearman": (
                    _build_rank_agreement_summary(decision_ead_spearman_by_date)
                ),
                "daily_decision_ead_abs_rank_kendall_tau": (
                    _build_rank_agreement_summary(decision_ead_kendall_by_date)
                ),
                "global_predictive_ead_abs_rank_spearman": (
                    compute_spearman_rank_correlation(
                        summary_shap["predictive_mean_abs_shap"].to_numpy(
                            dtype=float
                        ),
                        summary_shap["ead_decision_mean_abs_shap"].to_numpy(
                            dtype=float
                        ),
                    )
                ),
                "global_predictive_ead_abs_rank_kendall_tau": (
                    compute_kendall_tau_correlation(
                        summary_shap["predictive_mean_abs_shap"].to_numpy(
                            dtype=float
                        ),
                        summary_shap["ead_decision_mean_abs_shap"].to_numpy(
                            dtype=float
                        ),
                    )
                ),
                "global_decision_ead_abs_rank_spearman": (
                    compute_spearman_rank_correlation(
                        summary_shap["decision_mean_abs_shap"].to_numpy(dtype=float),
                        summary_shap["ead_decision_mean_abs_shap"].to_numpy(
                            dtype=float
                        ),
                    )
                ),
                "global_decision_ead_abs_rank_kendall_tau": (
                    compute_kendall_tau_correlation(
                        summary_shap["decision_mean_abs_shap"].to_numpy(dtype=float),
                        summary_shap["ead_decision_mean_abs_shap"].to_numpy(
                            dtype=float
                        ),
                    )
                ),
            }
        )
    return metrics


def _build_evaluation_metrics(
    daily_explanations: Sequence[DailyShapExplanation],
) -> dict[str, Any]:
    families = sorted(
        {
            family_name
            for explanation in daily_explanations
            for family_name in explanation.evaluation_metrics_by_family
        }
    )
    metrics_by_family: dict[str, Any] = {}
    for family_name in families:
        metric_names = sorted(
            {
                metric_name
                for explanation in daily_explanations
                for metric_name in explanation.evaluation_metrics_by_family[family_name]
            }
        )
        metrics_by_family[family_name] = {}
        for metric_name in metric_names:
            metric_summary = build_metric_summary(
                {
                    explanation.date: explanation.evaluation_metrics_by_family[family_name][
                        metric_name
                    ]
                    for explanation in daily_explanations
                }
            )
            if metric_name == "decision_deletion_auc":
                metric_summary["clipping"] = DECISION_DELETION_AUC_CLIPPING
                metric_summary["lower_is_better"] = True
                metric_summary["requires_positive_full_gain"] = True
                metric_summary["zero_if_all_strict_suffixes_nonpositive"] = True
            if metric_name == "decision_insertion_auc":
                metric_summary["clipping"] = DECISION_INSERTION_AUC_CLIPPING
                metric_summary["requires_positive_full_gain"] = True
                metric_summary["zero_if_all_strict_prefixes_nonpositive"] = True
            metrics_by_family[family_name][metric_name] = metric_summary
    return metrics_by_family


def _build_prediction_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_value_gains: Sequence[float] | None = None,
    actual_daily_regrets: Sequence[float] | None = None,
) -> dict[str, Any]:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must share the same shape for metric computation. "
            f"Got {y_true.shape} and {y_pred.shape}."
        )

    mse = float(mean_squared_error(y_true, y_pred))
    holdout_metrics: dict[str, Any] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "days": int(y_true.shape[0]),
        "targets_per_day": int(y_true.shape[1]) if y_true.ndim > 1 else 1,
        "predictions": int(y_true.size),
    }
    if decision_value_gains is not None:
        gains = np.asarray(decision_value_gains, dtype=float)
        holdout_metrics["mean_decision_value_gain"] = (
            float(np.mean(gains)) if gains.size > 0 else None
        )
    if actual_daily_regrets is not None:
        regrets = np.asarray(actual_daily_regrets, dtype=float)
        holdout_metrics["mean_actual_daily_regret"] = (
            float(np.mean(regrets)) if regrets.size > 0 else None
        )
    return {"holdout": holdout_metrics}


def _build_rank_agreement_summary(
    values_by_date: dict[str, float | None],
) -> dict[str, Any]:
    valid_values = [value for value in values_by_date.values() if value is not None]
    return {
        "values_by_date": values_by_date,
        "mean": float(np.mean(valid_values)) if valid_values else None,
        "median": float(np.median(valid_values)) if valid_values else None,
        "valid_days": len(valid_values),
        "total_days": len(values_by_date),
    }


def _descending_rank_map(values_by_feature: dict[str, float]) -> dict[str, int]:
    ranked = (
        pd.Series(values_by_feature, dtype=float)
        .rank(method="dense", ascending=False)
        .astype(int)
        .to_dict()
    )
    return {feature_name: int(rank) for feature_name, rank in ranked.items()}


def _resolve_observation_array(
    observation: pd.Series | Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    if isinstance(observation, pd.Series):
        ordered = observation.loc[list(feature_names)].to_numpy(dtype=float, copy=True)
        return ordered

    array = np.asarray(observation, dtype=float)
    if array.shape != (len(feature_names),):
        raise ValueError(
            "observation must have one value per feature. "
            f"Expected shape {(len(feature_names),)}, got {array.shape}."
        )
    return array


def _format_player_subset(
    subset: frozenset[int],
    player_names: Sequence[str],
) -> str:
    return "|".join(sorted(player_names[player_idx] for player_idx in subset))


def _evaluate_tree_expectation(
    tree: Any,
    observation: np.ndarray,
    coalition_mask: int,
) -> np.ndarray:
    features = tree.feature
    thresholds = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    weights = tree.weighted_n_node_samples
    values = np.asarray(tree.value[:, :, 0], dtype=float)
    leaf_indicator = -2
    memo: dict[tuple[int, int], np.ndarray] = {}

    def recurse(node_idx: int, active_mask: int) -> np.ndarray:
        memo_key = (node_idx, active_mask)
        if memo_key in memo:
            return memo[memo_key]

        feature_idx = int(features[node_idx])
        if feature_idx == leaf_indicator:
            result = values[node_idx].copy()
            memo[memo_key] = result
            return result

        left_child = int(children_left[node_idx])
        right_child = int(children_right[node_idx])
        if active_mask & (1 << feature_idx):
            if observation[feature_idx] <= thresholds[node_idx]:
                result = recurse(left_child, active_mask)
            else:
                result = recurse(right_child, active_mask)
            memo[memo_key] = result
            return result

        left_weight = float(weights[left_child])
        right_weight = float(weights[right_child])
        weight_sum = left_weight + right_weight
        if weight_sum <= 0:
            result = 0.5 * (
                recurse(left_child, active_mask) + recurse(right_child, active_mask)
            )
            memo[memo_key] = result
            return result

        result = (
            left_weight * recurse(left_child, active_mask)
            + right_weight * recurse(right_child, active_mask)
        ) / weight_sum
        memo[memo_key] = result
        return result

    return recurse(0, coalition_mask)


def _subset_weights(feature_count: int) -> tuple[float, ...]:
    denominator = math.factorial(feature_count)
    return tuple(
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / denominator
        for subset_size in range(feature_count)
    )
