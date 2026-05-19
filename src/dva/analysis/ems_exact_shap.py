from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from dva.analysis.evaluation_metrics import (
    build_attribution_ranking,
    build_metric_summary,
    compute_decision_activation_metrics,
    compute_decision_deletion_auc,
    compute_decision_insertion_auc,
    compute_exact_decision_infidelity,
    compute_rank_kendall_tau_from_rankings,
    compute_rank_spearman_from_rankings,
)
from dva.optimization import (
    DEFAULT_OPTIMIZATION_SOLVER,
    normalize_optimization_solver,
    pyomo_mip_gap,
    pyomo_solver_status,
    require_pyomo_solution,
    solve_pyomo_model,
)


DEFAULT_X_PATH = Path(
    "data/ems_data/processed/ems_zip_hour_features_2025_manhattan_wide_X.csv"
)
DEFAULT_Y_PATH = Path(
    "data/ems_data/processed/ems_zip_hour_features_2025_manhattan_wide_y.csv"
)
DEFAULT_METADATA_PATH = Path(
    "data/ems_data/processed/ems_zip_hour_features_2025_manhattan_wide_metadata.json"
)
DEFAULT_ZONE_ORDER_PATH = Path("data/ems_data/processed/ems_zip_wide_zone_order.csv")
DEFAULT_DISTANCE_MATRIX_PATH = Path(
    "data/ems_data/processed/ems_zip_centroid_distance_matrix_km.parquet"
)
DEFAULT_OUTPUT_DIR = Path("results/ems_exact_shap_case_study")
DEFAULT_HOLDOUT_HOURS = 24
DEFAULT_TEST_MONTHS = 1
DEFAULT_BACKGROUND_ROWS = 8
DEFAULT_COALITION_BATCH_SIZE = 64
DEFAULT_PROGRESS_EVERY_COALITIONS = 64
DEFAULT_COVERAGE_RADIUS_KM = 2.0
DEFAULT_FACILITY_BUDGET = 5
DEFAULT_OBJECTIVE_TOLERANCE = 1e-6
DEFAULT_CVAR_ALPHA = 0.9
DEFAULT_CVAR_SCENARIO_COUNT = 100
DEFAULT_EXCLUDED_ZIP_CODES = ("10468",)
EMS_COVERAGE_SOLVER_EXACT = "exact"
EMS_COVERAGE_SOLVER_NAIVE_GREEDY = "naive_greedy"
EMS_COVERAGE_SOLVER_GREEDY_MAX_COVER = "greedy_max_cover"
EMS_COVERAGE_SOLVER_LP_RELAXATION = "lp_relaxation"
EMS_COVERAGE_SOLVER_GUROBI = EMS_COVERAGE_SOLVER_EXACT
EMS_COVERAGE_SOLVER_GUROBI_LP_RELAXATION = EMS_COVERAGE_SOLVER_LP_RELAXATION
SUPPORTED_EMS_COVERAGE_SOLVERS = (
    EMS_COVERAGE_SOLVER_EXACT,
    EMS_COVERAGE_SOLVER_NAIVE_GREEDY,
    EMS_COVERAGE_SOLVER_GREEDY_MAX_COVER,
    EMS_COVERAGE_SOLVER_LP_RELAXATION,
)
DEFAULT_COVERAGE_SOLVER = EMS_COVERAGE_SOLVER_EXACT
_EMS_COVERAGE_SOLVER_ALIASES = {
    "exact": EMS_COVERAGE_SOLVER_EXACT,
    "pyomo": EMS_COVERAGE_SOLVER_EXACT,
    "gurobi": EMS_COVERAGE_SOLVER_EXACT,
    "naive_greedy": EMS_COVERAGE_SOLVER_NAIVE_GREEDY,
    "naive-greedy": EMS_COVERAGE_SOLVER_NAIVE_GREEDY,
    "greedy_max_cover": EMS_COVERAGE_SOLVER_GREEDY_MAX_COVER,
    "greedy-max-cover": EMS_COVERAGE_SOLVER_GREEDY_MAX_COVER,
    "gurobi_lp_relaxation": EMS_COVERAGE_SOLVER_LP_RELAXATION,
    "gurobi-lp-relaxation": EMS_COVERAGE_SOLVER_LP_RELAXATION,
    "linear_relaxation": EMS_COVERAGE_SOLVER_LP_RELAXATION,
    "linear-relaxation": EMS_COVERAGE_SOLVER_LP_RELAXATION,
    "lp": EMS_COVERAGE_SOLVER_LP_RELAXATION,
    "lp_relaxation": EMS_COVERAGE_SOLVER_LP_RELAXATION,
    "lp-relaxation": EMS_COVERAGE_SOLVER_LP_RELAXATION,
    "relaxation": EMS_COVERAGE_SOLVER_LP_RELAXATION,
}
EMS_TIMESTAMP_COLUMN = "timestamp_hour"
MONTH_FEATURE_COLUMN = "month"
_ZIP_SUFFIX_RE = re.compile(r"_zip_\d{5}$")
_HEURISTIC_SOLVER_STATUS = 0
_LP_RELAXATION_ROUNDING_TOLERANCE = 1e-9
EMS_DECISION_ACTIVATION_DEFINITION = "selected_facility_indices change"
_DECISION_PERMUTATION_SHAP_METHOD_ID = 1
_DECISION_KERNEL_SHAP_METHOD_ID = 2


@dataclass(frozen=True, slots=True)
class EmsFeatureGroup:
    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MaximumCoverageResult:
    objective_value: float
    covered_demand: float
    total_demand: float
    selected_facility_zip_codes: tuple[str, ...]
    covered_zip_codes: tuple[str, ...]
    selected_facility_indices: tuple[int, ...]
    covered_zone_indices: tuple[int, ...]
    solver_status: int | str
    optimal: bool
    mip_gap: float | None
    solver_name: str = DEFAULT_COVERAGE_SOLVER
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER
    solver_runtime_seconds: float | None = None
    risk_objective_value: float | None = None


@dataclass(frozen=True, slots=True)
class EmsExactShapConfig:
    x_path: Path = DEFAULT_X_PATH
    y_path: Path = DEFAULT_Y_PATH
    metadata_path: Path | None = DEFAULT_METADATA_PATH
    zone_order_path: Path = DEFAULT_ZONE_ORDER_PATH
    distance_matrix_path: Path = DEFAULT_DISTANCE_MATRIX_PATH
    outdir: Path = DEFAULT_OUTPUT_DIR
    holdout_hours: int = DEFAULT_HOLDOUT_HOURS
    test_months: int = DEFAULT_TEST_MONTHS
    max_hours: int | None = None
    background_rows: int = DEFAULT_BACKGROUND_ROWS
    coalition_batch_size: int = DEFAULT_COALITION_BATCH_SIZE
    progress_every_coalitions: int = DEFAULT_PROGRESS_EVERY_COALITIONS
    random_state: int = 0
    n_jobs: int = 1
    model_id: str = "xgb_001"
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_verbosity: int = 0
    train_sample_rows: int | None = None
    coverage_radius_km: float = DEFAULT_COVERAGE_RADIUS_KM
    facility_budget: int = DEFAULT_FACILITY_BUDGET
    solver_seed: int = 0
    mip_gap: float = 0.0
    mip_gap_abs: float = 1e-9
    gurobi_threads: int = 1
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE
    coverage_solver: str = DEFAULT_COVERAGE_SOLVER
    excluded_zip_codes: tuple[str, ...] = DEFAULT_EXCLUDED_ZIP_CODES
    save_coalition_values: bool = False
    compute_ante_decision_shap: bool = False
    cvar_alpha: float = DEFAULT_CVAR_ALPHA
    cvar_scenario_count: int = DEFAULT_CVAR_SCENARIO_COUNT
    compute_cvar_decision_shap: bool = True
    decision_permutation_shap_samples: tuple[int, ...] = ()
    decision_kernel_shap_samples: tuple[int, ...] = ()
    decision_permutation_shap_seed: int | None = None
    decision_kernel_shap_seed: int | None = None


@dataclass(frozen=True, slots=True)
class EmsExactShapOutputs:
    hourly_shap: pd.DataFrame
    predictive_zip_shap: pd.DataFrame
    coverage_solutions: pd.DataFrame
    summary_shap: pd.DataFrame
    prediction_metrics: dict[str, Any]
    evaluation_metrics: dict[str, Any]
    run_metadata: dict[str, Any]
    coalition_values: pd.DataFrame | None = None
    cvar_summary_shap: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class EmsTimeSplit:
    train_x: pd.DataFrame
    train_y: pd.DataFrame
    holdout_x: pd.DataFrame
    holdout_y: pd.DataFrame
    train_source_rows: tuple[int, ...]
    holdout_source_rows: tuple[int, ...]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp


class GroupedBackgroundCoalitionPredictor:
    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        feature_groups: Sequence[EmsFeatureGroup],
        background_frame: pd.DataFrame,
        *,
        coalition_batch_size: int,
    ) -> None:
        if coalition_batch_size <= 0:
            raise ValueError("coalition_batch_size must be strictly positive.")

        self.model = model
        self.feature_names = tuple(feature_names)
        self.feature_groups = tuple(feature_groups)
        self.player_names = tuple(group.name for group in self.feature_groups)
        self.player_count = len(self.feature_groups)
        self.coalition_count = 1 << self.player_count
        self.background_frame = background_frame.loc[:, list(self.feature_names)].copy()
        self.background_values = self.background_frame.to_numpy(dtype=float, copy=True)
        self.background_count = int(len(self.background_frame))
        self.coalition_batch_size = coalition_batch_size
        self._feature_index_by_name = {
            feature_name: feature_idx
            for feature_idx, feature_name in enumerate(self.feature_names)
        }
        self._included_indices_by_mask = tuple(
            self._feature_indices_for_mask(coalition_mask)
            for coalition_mask in range(self.coalition_count)
        )
        self.output_count = self._resolve_output_count()

    def predict_all_coalitions(
        self,
        observation: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
        *,
        progress_label: str | None = None,
        progress_every_coalitions: int = DEFAULT_PROGRESS_EVERY_COALITIONS,
    ) -> np.ndarray:
        if self.background_count <= 0:
            raise ValueError("Need at least one background row.")

        observation_values = _resolve_observation_array(observation, self.feature_names)
        predictions_by_mask = np.empty(
            (self.coalition_count, self.output_count),
            dtype=float,
        )

        for batch_start in range(
            0,
            self.coalition_count,
            self.coalition_batch_size,
        ):
            batch_end = min(
                self.coalition_count,
                batch_start + self.coalition_batch_size,
            )
            batch_masks = range(batch_start, batch_end)
            batch_values = np.tile(self.background_values, (len(batch_masks), 1))
            for local_idx, coalition_mask in enumerate(batch_masks):
                included_indices = self._included_indices_by_mask[coalition_mask]
                if included_indices.size == 0:
                    continue
                row_slice = slice(
                    local_idx * self.background_count,
                    (local_idx + 1) * self.background_count,
                )
                batch_values[row_slice, included_indices] = observation_values[
                    included_indices
                ]

            batch_predictions = np.asarray(
                self.model.predict(_as_float32_matrix(batch_values)),
                dtype=float,
            )
            if batch_predictions.ndim == 1:
                batch_predictions = batch_predictions[:, np.newaxis]
            batch_predictions = batch_predictions.reshape(
                len(batch_masks),
                self.background_count,
                self.output_count,
            )
            predictions_by_mask[batch_start:batch_end] = batch_predictions.mean(axis=1)
            if (
                progress_label is not None
                and progress_every_coalitions > 0
                and (
                    batch_end == self.coalition_count
                    or batch_end % progress_every_coalitions == 0
                )
            ):
                model_rows = batch_end * self.background_count
                print(
                    f"{progress_label}: predicted {batch_end:,}/"
                    f"{self.coalition_count:,} coalitions ({model_rows:,} XGB rows)",
                    flush=True,
                )

        return np.maximum(predictions_by_mask, 0.0)

    def _feature_indices_for_mask(self, coalition_mask: int) -> np.ndarray:
        indices: list[int] = []
        for player_idx, group in enumerate(self.feature_groups):
            if not coalition_mask & (1 << player_idx):
                continue
            indices.extend(self._feature_index_by_name[column] for column in group.columns)
        return np.asarray(indices, dtype=int)

    def _resolve_output_count(self) -> int:
        sample_prediction = np.asarray(
            self.model.predict(
                _frame_to_feature_matrix(self.background_frame.iloc[:1], self.feature_names)
            ),
            dtype=float,
        )
        if sample_prediction.ndim == 1:
            return 1
        return int(sample_prediction.shape[1])


def build_ems_feature_groups(
    feature_columns: Sequence[str],
) -> tuple[EmsFeatureGroup, ...]:
    groups: dict[str, list[str]] = {}
    for feature_name in feature_columns:
        if feature_name == MONTH_FEATURE_COLUMN:
            continue
        group_name = _ZIP_SUFFIX_RE.sub("", feature_name)
        groups.setdefault(group_name, []).append(feature_name)
    return tuple(
        EmsFeatureGroup(name=group_name, columns=tuple(columns))
        for group_name, columns in groups.items()
    )


def build_coverage_matrix(
    distance_matrix: pd.DataFrame,
    zip_codes: Sequence[str],
    *,
    coverage_radius_km: float,
) -> np.ndarray:
    if coverage_radius_km < 0:
        raise ValueError("coverage_radius_km must be non-negative.")

    normalized = _normalize_distance_matrix(distance_matrix)
    ordered_zip_codes = tuple(str(zip_code) for zip_code in zip_codes)
    missing_zip_codes = sorted(set(ordered_zip_codes) - set(normalized.index))
    missing_zip_codes.extend(
        sorted(set(ordered_zip_codes) - set(str(column) for column in normalized.columns))
    )
    if missing_zip_codes:
        raise ValueError(
            "Distance matrix is missing ZIP codes: "
            + ", ".join(sorted(set(missing_zip_codes)))
        )

    ordered_distances = normalized.loc[
        list(ordered_zip_codes),
        list(ordered_zip_codes),
    ].to_numpy(dtype=float, copy=True)
    return ordered_distances <= coverage_radius_km + 1e-9


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


def compute_permutation_shapley_values(
    coalition_values: Sequence[float] | np.ndarray,
    feature_count: int,
    *,
    sample_count: int,
    random_state: int | Sequence[int] | np.random.SeedSequence | np.random.Generator,
) -> np.ndarray:
    coalition_array = _validate_scalar_approximation_game(
        coalition_values,
        feature_count,
        "permutation SHAP",
    )
    sample_count = _validate_positive_sample_count(
        sample_count,
        "sample_count",
    )
    if feature_count <= 0:
        raise ValueError("feature_count must be strictly positive.")

    rng = np.random.default_rng(random_state)
    shap_values = np.zeros(feature_count, dtype=float)
    for _ in range(sample_count):
        coalition_mask = 0
        previous_value = float(coalition_array[coalition_mask])
        for feature_idx in rng.permutation(feature_count):
            coalition_with_feature = coalition_mask | (1 << int(feature_idx))
            next_value = float(coalition_array[coalition_with_feature])
            shap_values[int(feature_idx)] += next_value - previous_value
            coalition_mask = coalition_with_feature
            previous_value = next_value
    return shap_values / float(sample_count)


def compute_kernel_shapley_values(
    coalition_values: Sequence[float] | np.ndarray,
    feature_count: int,
    *,
    sample_count: int,
    random_state: int | Sequence[int] | np.random.SeedSequence | np.random.Generator,
) -> np.ndarray:
    coalition_array = _validate_scalar_approximation_game(
        coalition_values,
        feature_count,
        "kernel SHAP",
    )
    sample_count = _validate_positive_sample_count(
        sample_count,
        "sample_count",
    )
    if feature_count <= 0:
        raise ValueError("feature_count must be strictly positive.")
    if feature_count == 1:
        return np.array([float(coalition_array[-1] - coalition_array[0])], dtype=float)

    full_mask = (1 << feature_count) - 1
    interior_masks = np.arange(1, full_mask, dtype=int)
    centered_values = coalition_array - float(coalition_array[0])
    full_value = float(centered_values[full_mask])
    rng = np.random.default_rng(random_state)
    sampled_masks = rng.choice(
        interior_masks,
        size=sample_count,
        replace=sample_count > len(interior_masks),
    )
    design = np.zeros((sample_count, feature_count), dtype=float)
    weights = np.empty(sample_count, dtype=float)
    for row_idx, coalition_mask in enumerate(sampled_masks):
        mask = int(coalition_mask)
        subset_size = mask.bit_count()
        weights[row_idx] = _shapley_kernel_weight(feature_count, subset_size)
        for feature_idx in range(feature_count):
            if mask & (1 << feature_idx):
                design[row_idx, feature_idx] = 1.0

    last_column = design[:, -1]
    reduced_design = design[:, :-1] - last_column[:, np.newaxis]
    adjusted_values = centered_values[sampled_masks] - last_column * full_value
    sqrt_weights = np.sqrt(weights)
    theta, *_ = np.linalg.lstsq(
        reduced_design * sqrt_weights[:, np.newaxis],
        adjusted_values * sqrt_weights,
        rcond=None,
    )
    shap_values = np.empty(feature_count, dtype=float)
    shap_values[:-1] = theta
    shap_values[-1] = full_value - float(theta.sum())
    return shap_values


def normalize_ems_coverage_solver(solver_name: str) -> str:
    solver_key = str(solver_name).strip().lower().replace(" ", "_")
    normalized = _EMS_COVERAGE_SOLVER_ALIASES.get(solver_key)
    if normalized is None:
        raise ValueError(
            "Unsupported EMS coverage solver: "
            f"{solver_name!r}. Expected one of "
            + ", ".join(SUPPORTED_EMS_COVERAGE_SOLVERS)
            + "."
        )
    return normalized


def solve_ems_coverage(
    demand: Sequence[float] | np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    *,
    facility_budget: int,
    solver_name: str = DEFAULT_COVERAGE_SOLVER,
    name: str = "ems_coverage",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER,
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE,
) -> MaximumCoverageResult:
    solver = normalize_ems_coverage_solver(solver_name)
    if solver == EMS_COVERAGE_SOLVER_EXACT:
        return solve_maximum_coverage(
            demand,
            coverage_matrix,
            zip_codes,
            facility_budget=facility_budget,
            name=name,
            log_to_console=log_to_console,
            solver_params=solver_params,
            optimization_solver=optimization_solver,
            objective_tolerance=objective_tolerance,
        )
    if solver == EMS_COVERAGE_SOLVER_NAIVE_GREEDY:
        return solve_naive_greedy_coverage(
            demand,
            coverage_matrix,
            zip_codes,
            facility_budget=facility_budget,
        )
    if solver == EMS_COVERAGE_SOLVER_GREEDY_MAX_COVER:
        return solve_greedy_max_cover_coverage(
            demand,
            coverage_matrix,
            zip_codes,
            facility_budget=facility_budget,
        )
    if solver == EMS_COVERAGE_SOLVER_LP_RELAXATION:
        return solve_gurobi_lp_relaxation_coverage(
            demand,
            coverage_matrix,
            zip_codes,
            facility_budget=facility_budget,
            name=name,
            log_to_console=log_to_console,
            solver_params=solver_params,
            optimization_solver=optimization_solver,
            objective_tolerance=objective_tolerance,
        )
    raise AssertionError(f"Unhandled EMS coverage solver: {solver}")


def solve_naive_greedy_coverage(
    demand: Sequence[float] | np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    *,
    facility_budget: int,
) -> MaximumCoverageResult:
    demand_array, coverage_array, ordered_zip_codes = _prepare_coverage_inputs(
        demand,
        coverage_matrix,
        zip_codes,
        facility_budget=facility_budget,
    )
    _validate_zip_aligned_demand(demand_array, ordered_zip_codes, "naive_greedy")

    selected_indices = tuple(
        sorted(
            _rank_indices_descending(demand_array)[
                : min(int(facility_budget), len(ordered_zip_codes))
            ]
        )
    )
    return _build_coverage_result(
        demand_array=demand_array,
        coverage_array=coverage_array,
        ordered_zip_codes=ordered_zip_codes,
        selected_indices=selected_indices,
        solver_status=_HEURISTIC_SOLVER_STATUS,
        optimal=False,
        mip_gap=None,
        solver_name=EMS_COVERAGE_SOLVER_NAIVE_GREEDY,
    )


def solve_greedy_max_cover_coverage(
    demand: Sequence[float] | np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    *,
    facility_budget: int,
) -> MaximumCoverageResult:
    demand_array, coverage_array, ordered_zip_codes = _prepare_coverage_inputs(
        demand,
        coverage_matrix,
        zip_codes,
        facility_budget=facility_budget,
    )
    _validate_zip_aligned_demand(demand_array, ordered_zip_codes, "greedy_max_cover")

    demand_count, facility_count = coverage_array.shape
    covered_mask = np.zeros(demand_count, dtype=bool)
    selected_order: list[int] = []
    max_facilities = min(int(facility_budget), facility_count)
    for _ in range(max_facilities):
        best_facility_idx: int | None = None
        best_gain = -math.inf
        for facility_idx in range(facility_count):
            if facility_idx in selected_order:
                continue
            newly_covered = coverage_array[:, facility_idx] & ~covered_mask
            gain = float(demand_array[newly_covered].sum())
            if best_facility_idx is None or gain > best_gain + 1e-12:
                best_facility_idx = facility_idx
                best_gain = gain
            elif math.isclose(gain, best_gain, rel_tol=0.0, abs_tol=1e-12):
                best_facility_idx = min(best_facility_idx, facility_idx)

        if best_facility_idx is None:
            break
        selected_order.append(best_facility_idx)
        covered_mask |= coverage_array[:, best_facility_idx]

    return _build_coverage_result(
        demand_array=demand_array,
        coverage_array=coverage_array,
        ordered_zip_codes=ordered_zip_codes,
        selected_indices=tuple(sorted(selected_order)),
        solver_status=_HEURISTIC_SOLVER_STATUS,
        optimal=False,
        mip_gap=None,
        solver_name=EMS_COVERAGE_SOLVER_GREEDY_MAX_COVER,
    )


def solve_gurobi_lp_relaxation_coverage(
    demand: Sequence[float] | np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    *,
    facility_budget: int,
    name: str = "ems_lp_relaxation_coverage",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER,
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE,
) -> MaximumCoverageResult:
    return solve_lp_relaxation_coverage(
        demand,
        coverage_matrix,
        zip_codes,
        facility_budget=facility_budget,
        name=name,
        log_to_console=log_to_console,
        solver_params=solver_params,
        optimization_solver=optimization_solver,
        objective_tolerance=objective_tolerance,
    )


def solve_lp_relaxation_coverage(
    demand: Sequence[float] | np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    *,
    facility_budget: int,
    name: str = "ems_lp_relaxation_coverage",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER,
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE,
) -> MaximumCoverageResult:
    demand_array, coverage_array, ordered_zip_codes = _prepare_coverage_inputs(
        demand,
        coverage_matrix,
        zip_codes,
        facility_budget=facility_budget,
    )
    if objective_tolerance < 0:
        raise ValueError("objective_tolerance must be non-negative.")

    demand_count, facility_count = coverage_array.shape
    model = _build_maximum_coverage_pyomo_model(
        name=name,
        demand_array=demand_array,
        coverage_array=coverage_array,
        facility_budget=facility_budget,
        relax_integrality=True,
    )
    x = model.x
    y = model.y
    primary_objective = model.primary_objective_expression
    primary_solve_result = _solve_coverage_model(
        model,
        label="LP-relaxed coverage model",
        solver_params=solver_params,
        optimization_solver=optimization_solver,
        log_to_console=log_to_console,
    )
    primary_objective_value = float(pyo.value(primary_objective))

    model.deterministic_primary_objective_floor = pyo.Constraint(
        expr=primary_objective >= primary_objective_value - objective_tolerance
    )
    model.objective.deactivate()
    model.tie_break_objective = pyo.Objective(
        expr=_build_deterministic_coverage_tie_break_expression(
            x=x,
            y=y,
            demand_count=demand_count,
            facility_count=facility_count,
        ),
        sense=pyo.maximize,
    )
    tie_break_solve_result = _solve_coverage_model(
        model,
        label="LP-relaxed coverage tie-break model",
        solver_params=solver_params,
        optimization_solver=optimization_solver,
        log_to_console=log_to_console,
    )

    selected_indices = _round_lp_relaxation_facility_indices(
        np.array([float(pyo.value(x[facility_idx])) for facility_idx in range(facility_count)]),
        facility_budget=facility_budget,
    )
    return _build_coverage_result(
        demand_array=demand_array,
        coverage_array=coverage_array,
        ordered_zip_codes=ordered_zip_codes,
        selected_indices=selected_indices,
        solver_status=pyomo_solver_status(tie_break_solve_result),
        optimal=False,
        mip_gap=None,
        solver_name=EMS_COVERAGE_SOLVER_LP_RELAXATION,
        optimization_solver=normalize_optimization_solver(optimization_solver),
        solver_runtime_seconds=_sum_solver_runtime_seconds(
            primary_solve_result,
            tie_break_solve_result,
        ),
    )


def solve_maximum_coverage(
    demand: Sequence[float] | np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    *,
    facility_budget: int,
    name: str = "ems_maximum_coverage",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER,
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE,
) -> MaximumCoverageResult:
    demand_array, coverage_array, ordered_zip_codes = _prepare_coverage_inputs(
        demand,
        coverage_matrix,
        zip_codes,
        facility_budget=facility_budget,
    )
    if objective_tolerance < 0:
        raise ValueError("objective_tolerance must be non-negative.")

    demand_count, facility_count = coverage_array.shape
    model = _build_maximum_coverage_pyomo_model(
        name=name,
        demand_array=demand_array,
        coverage_array=coverage_array,
        facility_budget=facility_budget,
        relax_integrality=False,
    )
    x = model.x
    y = model.y
    primary_objective = model.primary_objective_expression
    primary_solve_result = _solve_coverage_model(
        model,
        label="Maximum coverage model",
        solver_params=solver_params,
        optimization_solver=optimization_solver,
        log_to_console=log_to_console,
    )
    primary_objective_value = float(pyo.value(primary_objective))

    model.deterministic_primary_objective_floor = pyo.Constraint(
        expr=primary_objective >= primary_objective_value - objective_tolerance
    )
    model.objective.deactivate()
    model.tie_break_objective = pyo.Objective(
        expr=_build_deterministic_coverage_tie_break_expression(
            x=x,
            y=y,
            demand_count=demand_count,
            facility_count=facility_count,
        ),
        sense=pyo.maximize,
    )
    tie_break_solve_result = _solve_coverage_model(
        model,
        label="Maximum coverage tie-break model",
        solver_params=solver_params,
        optimization_solver=optimization_solver,
        log_to_console=log_to_console,
    )

    selected_indices = tuple(
        facility_idx
        for facility_idx in range(facility_count)
        if float(pyo.value(x[facility_idx])) > 0.5
    )
    return _build_coverage_result(
        demand_array=demand_array,
        coverage_array=coverage_array,
        ordered_zip_codes=ordered_zip_codes,
        selected_indices=selected_indices,
        solver_status=pyomo_solver_status(tie_break_solve_result),
        optimal=tie_break_solve_result.optimal,
        mip_gap=(
            None
            if tie_break_solve_result.optimal
            else pyomo_mip_gap(tie_break_solve_result)
        ),
        solver_name=EMS_COVERAGE_SOLVER_EXACT,
        optimization_solver=normalize_optimization_solver(optimization_solver),
        solver_runtime_seconds=_sum_solver_runtime_seconds(
            primary_solve_result,
            tie_break_solve_result,
        ),
    )


def solve_cvar_coverage(
    demand_scenarios: np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    *,
    facility_budget: int,
    alpha: float = DEFAULT_CVAR_ALPHA,
    name: str = "ems_cvar_coverage",
    log_to_console: bool = False,
    solver_params: Mapping[str, float | int | str] | None = None,
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER,
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE,
) -> MaximumCoverageResult:
    scenarios = np.maximum(np.asarray(demand_scenarios, dtype=float), 0.0)
    if scenarios.ndim != 2:
        raise ValueError("demand_scenarios must have shape (scenario_count, demand_count).")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1).")
    if objective_tolerance < 0:
        raise ValueError("objective_tolerance must be non-negative.")

    scenario_count, demand_count = scenarios.shape
    if scenario_count <= 0:
        raise ValueError("demand_scenarios must include at least one scenario.")
    coverage_array = np.asarray(coverage_matrix, dtype=bool)
    ordered_zip_codes = tuple(str(zip_code) for zip_code in zip_codes)
    facility_count = len(ordered_zip_codes)

    if coverage_array.shape != (demand_count, facility_count):
        raise ValueError(
            "coverage_matrix must have shape (demand_count, facility_count). "
            f"Got {coverage_array.shape}, expected {(demand_count, facility_count)}."
        )
    if facility_budget < 0:
        raise ValueError("facility_budget must be non-negative.")

    model = pyo.ConcreteModel(name=name)
    model.FACILITIES = pyo.Set(initialize=range(facility_count))
    model.DEMANDS = pyo.Set(initialize=range(demand_count))
    model.SCENARIOS = pyo.Set(initialize=range(scenario_count))
    model.x = pyo.Var(model.FACILITIES, domain=pyo.Binary)
    model.y = pyo.Var(model.DEMANDS, domain=pyo.Binary)
    model.eta = pyo.Var(domain=pyo.Reals)
    model.shortfall = pyo.Var(model.SCENARIOS, domain=pyo.NonNegativeReals)
    x = model.x
    y = model.y
    shortfall = model.shortfall
    scenario_totals = scenarios.sum(axis=1)

    def coverage_rule(coverage_model: pyo.ConcreteModel, demand_idx: int) -> pyo.Expression:
        covering_facilities = [
            facility_idx
            for facility_idx in range(facility_count)
            if coverage_array[demand_idx, facility_idx]
        ]
        return coverage_model.y[demand_idx] <= sum(
            coverage_model.x[facility_idx] for facility_idx in covering_facilities
        )

    model.coverage = pyo.Constraint(model.DEMANDS, rule=coverage_rule)
    model.facility_budget = pyo.Constraint(
        expr=sum(x[facility_idx] for facility_idx in range(facility_count))
        <= int(facility_budget)
    )

    def scenario_reward(scenario_idx: int) -> pyo.Expression:
        return sum(
            _coverage_weight(
                float(scenarios[scenario_idx, demand_idx]),
                float(scenario_totals[scenario_idx]),
            )
            * y[demand_idx]
            for demand_idx in range(demand_count)
        )

    model.cvar_shortfall = pyo.Constraint(
        model.SCENARIOS,
        rule=lambda coverage_model, scenario_idx: (
            shortfall[scenario_idx]
            >= coverage_model.eta - scenario_reward(int(scenario_idx))
        ),
    )

    model.cvar_objective_expression = pyo.Expression(
        expr=model.eta - (
        1.0 / ((1.0 - alpha) * float(scenario_count))
        ) * sum(shortfall[scenario_idx] for scenario_idx in range(scenario_count))
    )

    model.objective = pyo.Objective(
        expr=model.cvar_objective_expression,
        sense=pyo.maximize,
    )
    primary_solve_result = _solve_coverage_model(
        model,
        label="CVaR coverage model",
        solver_params=solver_params,
        optimization_solver=optimization_solver,
        log_to_console=log_to_console,
    )

    cvar_objective_value = float(pyo.value(model.cvar_objective_expression))
    model.cvar_primary_objective_floor = pyo.Constraint(
        expr=model.cvar_objective_expression >= cvar_objective_value - objective_tolerance
    )
    model.objective.deactivate()
    model.tie_break_objective = pyo.Objective(
        expr=_build_deterministic_coverage_tie_break_expression(
            x=x,
            y=y,
            demand_count=demand_count,
            facility_count=facility_count,
        ),
        sense=pyo.maximize,
    )
    tie_break_solve_result = _solve_coverage_model(
        model,
        label="CVaR coverage tie-break model",
        solver_params=solver_params,
        optimization_solver=optimization_solver,
        log_to_console=log_to_console,
    )

    selected_indices = tuple(
        facility_idx
        for facility_idx in range(facility_count)
        if float(pyo.value(x[facility_idx])) > 0.5
    )

    return _build_coverage_result(
        demand_array=scenarios.mean(axis=0),
        coverage_array=coverage_array,
        ordered_zip_codes=ordered_zip_codes,
        selected_indices=selected_indices,
        solver_status=pyomo_solver_status(tie_break_solve_result),
        optimal=tie_break_solve_result.optimal,
        mip_gap=(
            None
            if tie_break_solve_result.optimal
            else pyomo_mip_gap(tie_break_solve_result)
        ),
        solver_name="cvar",
        optimization_solver=normalize_optimization_solver(optimization_solver),
        solver_runtime_seconds=_sum_solver_runtime_seconds(
            primary_solve_result,
            tie_break_solve_result,
        ),
        risk_objective_value=cvar_objective_value,
    )


def run_ems_exact_shap(config: EmsExactShapConfig) -> EmsExactShapOutputs:
    _validate_config(config)
    started_at = time.perf_counter()
    coverage_solver = normalize_ems_coverage_solver(config.coverage_solver)
    decision_permutation_sample_counts = _normalize_decision_shap_sample_counts(
        config.decision_permutation_shap_samples,
        "decision_permutation_shap_samples",
    )
    decision_kernel_sample_counts = _normalize_decision_shap_sample_counts(
        config.decision_kernel_shap_samples,
        "decision_kernel_shap_samples",
    )
    decision_permutation_seed = _resolve_decision_shap_seed(
        config.decision_permutation_shap_seed,
        config.random_state,
        "decision_permutation_shap_seed",
    )
    decision_kernel_seed = _resolve_decision_shap_seed(
        config.decision_kernel_shap_seed,
        config.random_state,
        "decision_kernel_shap_seed",
    )

    x_frame, y_frame, metadata = _load_ems_frames(config)
    zone_order = _load_zone_order(config.zone_order_path, tuple(_target_columns(y_frame)))
    target_columns = tuple(zone_order["target_column"].astype(str))
    zip_codes = tuple(zone_order["zip_code"].astype(str))
    y_frame = y_frame.loc[:, [EMS_TIMESTAMP_COLUMN, *target_columns]].copy()

    feature_columns = _resolve_feature_columns(x_frame, metadata)
    feature_groups = build_ems_feature_groups(feature_columns)
    player_names = tuple(group.name for group in feature_groups)
    time_split = _build_time_split(
        x_frame.loc[:, [EMS_TIMESTAMP_COLUMN, *feature_columns]],
        y_frame,
        test_months=config.test_months,
    )
    train_x = time_split.train_x
    train_y = time_split.train_y
    holdout_x = time_split.holdout_x
    holdout_y = time_split.holdout_y
    unsampled_train_rows = len(train_x)
    train_x, train_y = _maybe_sample_training_rows(
        train_x,
        train_y,
        train_sample_rows=config.train_sample_rows,
        random_state=config.random_state,
    )

    explain_x, explain_y, explained_rows = _sample_explanation_hours(
        holdout_x,
        holdout_y,
        time_split.holdout_source_rows,
        holdout_hours=config.holdout_hours,
        max_hours=config.max_hours,
        random_state=config.random_state,
    )
    if explain_x.empty:
        raise ValueError("No explanation hours were sampled from the holdout set.")

    print(
        f"Using final {config.test_months} calendar month(s) as holdout/test set "
        f"({len(holdout_x):,} hours from {time_split.holdout_start} "
        f"through {time_split.holdout_end}); explaining {len(explain_x):,} "
        "sampled holdout hours.",
        flush=True,
    )

    print(
        "Training XGBRegressor "
        f"on {len(train_x):,} hours with {len(feature_columns)} raw features "
        f"grouped into {len(player_names)} players.",
        flush=True,
    )
    training_started_at = time.perf_counter()
    model = _fit_xgb_regressor(
        train_frame=train_x,
        y_train=train_y.loc[:, list(target_columns)],
        feature_columns=feature_columns,
        config=config,
    )
    print(
        f"Finished XGBRegressor training in "
        f"{time.perf_counter() - training_started_at:.2f}s.",
        flush=True,
    )
    residual_matrix = _build_training_residual_matrix(
        model=model,
        train_frame=train_x,
        train_y=train_y,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )

    holdout_predictions = np.maximum(
        np.asarray(
            model.predict(_frame_to_feature_matrix(holdout_x, feature_columns)),
            dtype=float,
        ),
        0.0,
    )
    if holdout_predictions.ndim == 1:
        holdout_predictions = holdout_predictions[:, np.newaxis]
    prediction_metrics = _build_prediction_metrics(
        y_true=holdout_y.loc[:, list(target_columns)].to_numpy(dtype=float, copy=True),
        y_pred=holdout_predictions,
    )

    background_frame = _sample_background_frame(
        train_frame=train_x,
        feature_columns=feature_columns,
        rows=config.background_rows,
        random_state=config.random_state,
    )
    coalition_predictor = GroupedBackgroundCoalitionPredictor(
        model=model,
        feature_names=feature_columns,
        feature_groups=feature_groups,
        background_frame=background_frame,
        coalition_batch_size=config.coalition_batch_size,
    )
    distance_matrix = _load_distance_matrix(config.distance_matrix_path)
    coverage_matrix = build_coverage_matrix(
        distance_matrix,
        zip_codes,
        coverage_radius_km=config.coverage_radius_km,
    )

    hourly_rows: list[dict[str, Any]] = []
    predictive_zip_rows: list[dict[str, Any]] = []
    coverage_solution_rows: list[dict[str, Any]] = []
    coalition_rows: list[dict[str, Any]] = []
    metric_values: dict[str, dict[str, float | None]] = {
        "predictive_decision_deletion_auc": {},
        "predictive_decision_insertion_auc": {},
        "predictive_decision_infidelity": {},
        "decision_decision_deletion_auc": {},
        "decision_decision_insertion_auc": {},
        "decision_decision_infidelity": {},
        "abs_rank_spearman": {},
        "abs_rank_kendall_tau": {},
    }

    explain_timestamps = tuple(explain_x[EMS_TIMESTAMP_COLUMN])
    for hour_idx, timestamp_hour in enumerate(explain_timestamps, start=1):
        hour_started_at = time.perf_counter()
        timestamp_key = str(timestamp_hour)
        true_demand = explain_y.loc[
            hour_idx - 1,
            list(target_columns),
        ].to_numpy(dtype=float, copy=True)
        coalition_predictions = coalition_predictor.predict_all_coalitions(
            explain_x.loc[hour_idx - 1, list(feature_columns)],
            progress_label=(
                f"[{hour_idx}/{len(explain_timestamps)}] predictive SHAP "
                f"{timestamp_key}"
            ),
            progress_every_coalitions=config.progress_every_coalitions,
        )
        predictive_values = coalition_predictions.sum(axis=1)
        (
            decision_values,
            baseline_solution,
            full_solution,
            decision_solutions,
        ) = _solve_decision_values(
            coalition_demand_matrix=coalition_predictions,
            true_demand=true_demand,
            coverage_matrix=coverage_matrix,
            zip_codes=zip_codes,
            config=config,
            coverage_solver=coverage_solver,
            progress_label=(
                f"[{hour_idx}/{len(explain_timestamps)}] decision SHAP "
                f"{timestamp_key}"
            ),
        )
        oracle_solution = solve_maximum_coverage(
            true_demand,
            coverage_matrix,
            zip_codes,
            facility_budget=config.facility_budget,
            name=f"ems_max_coverage_oracle_{hour_idx}",
            solver_params=_build_solver_params(config),
            optimization_solver=config.optimization_solver,
            objective_tolerance=config.objective_tolerance,
        )
        oracle_value = _realized_coverage_value(
            oracle_solution.covered_zone_indices,
            true_demand,
        )

        cvar_decision_values = None
        cvar_baseline_solution = None
        cvar_full_solution = None
        cvar_decision_shap = None
        cvar_decision_characteristic_values = None
        cvar_decision_activation = None
        cvar_decision_solutions = None
        if config.compute_cvar_decision_shap:
            residual_scenarios = _sample_residual_scenarios(
                residual_matrix,
                scenario_count=config.cvar_scenario_count,
                random_state=config.random_state + hour_idx,
            )
            coalition_demand_scenarios = _build_coalition_demand_scenarios(
                coalition_predictions,
                residual_scenarios,
            )
            (
                cvar_decision_values,
                cvar_baseline_solution,
                cvar_full_solution,
                cvar_decision_solutions,
            ) = _solve_cvar_decision_values(
                coalition_demand_scenarios=coalition_demand_scenarios,
                true_demand=true_demand,
                coverage_matrix=coverage_matrix,
                zip_codes=zip_codes,
                config=config,
                progress_label=(
                    f"[{hour_idx}/{len(explain_timestamps)}] CVaR decision SHAP "
                    f"{timestamp_key}"
                ),
            )
            cvar_decision_characteristic_values = (
                cvar_decision_values - cvar_decision_values[0]
            )
            cvar_decision_shap = compute_exact_shapley_values(
                cvar_decision_characteristic_values,
                feature_count=len(player_names),
            )
            cvar_decision_activation = compute_decision_activation_metrics(
                cvar_decision_characteristic_values,
                cvar_decision_solutions,
                len(player_names),
                _maximum_coverage_decision_changed,
            )

        predictive_zip_shap = compute_exact_shapley_values(
            coalition_predictions,
            feature_count=len(player_names),
        )
        predictive_shap = predictive_zip_shap.sum(axis=1)
        decision_characteristic_values = decision_values - decision_values[0]
        decision_shap = compute_exact_shapley_values(
            decision_characteristic_values,
            feature_count=len(player_names),
        )
        decision_permutation_shap_by_samples = {
            sample_count: compute_permutation_shapley_values(
                decision_characteristic_values,
                len(player_names),
                sample_count=sample_count,
                random_state=_decision_shap_seed_sequence(
                    decision_permutation_seed,
                    hour_idx,
                    _DECISION_PERMUTATION_SHAP_METHOD_ID,
                    sample_count,
                ),
            )
            for sample_count in decision_permutation_sample_counts
        }
        decision_kernel_shap_by_samples = {
            sample_count: compute_kernel_shapley_values(
                decision_characteristic_values,
                len(player_names),
                sample_count=sample_count,
                random_state=_decision_shap_seed_sequence(
                    decision_kernel_seed,
                    hour_idx,
                    _DECISION_KERNEL_SHAP_METHOD_ID,
                    sample_count,
                ),
            )
            for sample_count in decision_kernel_sample_counts
        }
        decision_activation = compute_decision_activation_metrics(
            decision_characteristic_values,
            decision_solutions,
            len(player_names),
            _maximum_coverage_decision_changed,
        )
        ante_decision_values = None
        ante_decision_characteristic_values = None
        ante_decision_shap = None
        if config.compute_ante_decision_shap:
            ante_reference_demand = coalition_predictions[-1]
            ante_decision_values = np.asarray(
                [
                    _realized_coverage_value(
                        solution.covered_zone_indices,
                        ante_reference_demand,
                    )
                    for solution in decision_solutions
                ],
                dtype=float,
            )
            ante_decision_characteristic_values = (
                ante_decision_values - ante_decision_values[0]
            )
            ante_decision_shap = compute_exact_shapley_values(
                ante_decision_characteristic_values,
                feature_count=len(player_names),
            )
        predictive_ranking = build_attribution_ranking(predictive_shap, player_names)
        decision_ranking = build_attribution_ranking(decision_shap, player_names)
        abs_rank_spearman = compute_rank_spearman_from_rankings(
            predictive_ranking,
            decision_ranking,
        )
        abs_rank_kendall_tau = compute_rank_kendall_tau_from_rankings(
            predictive_ranking,
            decision_ranking,
        )
        predictive_auc = compute_decision_insertion_auc(
            predictive_shap,
            decision_characteristic_values,
            player_names,
        )
        predictive_deletion_auc = compute_decision_deletion_auc(
            predictive_shap,
            decision_characteristic_values,
            player_names,
        )
        predictive_infidelity = compute_exact_decision_infidelity(
            predictive_shap,
            decision_characteristic_values,
            player_names,
        )
        decision_auc = compute_decision_insertion_auc(
            decision_shap,
            decision_characteristic_values,
            player_names,
        )
        decision_deletion_auc = compute_decision_deletion_auc(
            decision_shap,
            decision_characteristic_values,
            player_names,
        )
        decision_infidelity = compute_exact_decision_infidelity(
            decision_shap,
            decision_characteristic_values,
            player_names,
        )
        metric_values["predictive_decision_deletion_auc"][
            timestamp_key
        ] = predictive_deletion_auc
        metric_values["predictive_decision_insertion_auc"][
            timestamp_key
        ] = predictive_auc
        metric_values["predictive_decision_infidelity"][
            timestamp_key
        ] = predictive_infidelity
        metric_values["decision_decision_deletion_auc"][
            timestamp_key
        ] = decision_deletion_auc
        metric_values["decision_decision_insertion_auc"][timestamp_key] = decision_auc
        metric_values["decision_decision_infidelity"][timestamp_key] = decision_infidelity
        metric_values["abs_rank_spearman"][timestamp_key] = abs_rank_spearman
        metric_values["abs_rank_kendall_tau"][timestamp_key] = abs_rank_kendall_tau

        actual_total_demand = _total_demand(true_demand)
        baseline_covered_demand = _covered_demand_value(
            baseline_solution.covered_zone_indices,
            true_demand,
        )
        full_covered_demand = _covered_demand_value(
            full_solution.covered_zone_indices,
            true_demand,
        )
        oracle_covered_demand = _covered_demand_value(
            oracle_solution.covered_zone_indices,
            true_demand,
        )
        row: dict[str, Any] = {
            "timestamp_hour": timestamp_key,
            "zip_count": len(zip_codes),
            "player_count": len(player_names),
            "coalition_count": coalition_predictor.coalition_count,
            "background_rows": coalition_predictor.background_count,
            "coverage_radius_km": float(config.coverage_radius_km),
            "facility_budget": int(config.facility_budget),
            "predictive_baseline_total": float(predictive_values[0]),
            "predictive_full_total": float(predictive_values[-1]),
            "predictive_total_gain": float(predictive_values[-1] - predictive_values[0]),
            "actual_total_demand": actual_total_demand,
            "decision_baseline_value": float(decision_values[0]),
            "decision_full_value": float(decision_values[-1]),
            "decision_value_gain": float(decision_characteristic_values[-1]),
            "oracle_value": float(oracle_value),
            "actual_regret": float(oracle_value - decision_values[-1]),
            "decision_baseline_covered_demand": baseline_covered_demand,
            "decision_full_covered_demand": full_covered_demand,
            "oracle_covered_demand": oracle_covered_demand,
            "baseline_selected_zip_codes": json.dumps(
                list(baseline_solution.selected_facility_zip_codes)
            ),
            "full_selected_zip_codes": json.dumps(
                list(full_solution.selected_facility_zip_codes)
            ),
            "oracle_selected_zip_codes": json.dumps(
                list(oracle_solution.selected_facility_zip_codes)
            ),
            "baseline_covered_zip_codes": json.dumps(
                list(baseline_solution.covered_zip_codes)
            ),
            "full_covered_zip_codes": json.dumps(list(full_solution.covered_zip_codes)),
            "oracle_covered_zip_codes": json.dumps(
                list(oracle_solution.covered_zip_codes)
            ),
            "predictive_decision_deletion_auc": predictive_deletion_auc,
            "predictive_decision_insertion_auc": predictive_auc,
            "predictive_decision_infidelity": predictive_infidelity,
            "decision_decision_deletion_auc": decision_deletion_auc,
            "decision_decision_insertion_auc": decision_auc,
            "decision_decision_infidelity": decision_infidelity,
            "abs_rank_spearman": abs_rank_spearman,
            "abs_rank_kendall_tau": abs_rank_kendall_tau,
            "hour_runtime_seconds": time.perf_counter() - hour_started_at,
        }
        for player_name, shap_value in zip(player_names, predictive_shap, strict=True):
            row[f"predictive_shap_{player_name}"] = float(shap_value)
        for player_name, shap_value in zip(player_names, decision_shap, strict=True):
            row[f"decision_shap_{player_name}"] = float(shap_value)
        for sample_count, shap_values in decision_permutation_shap_by_samples.items():
            for player_name, shap_value in zip(player_names, shap_values, strict=True):
                row[f"decision_permutation_shap_{sample_count}_{player_name}"] = float(
                    shap_value
                )
        for sample_count, shap_values in decision_kernel_shap_by_samples.items():
            for player_name, shap_value in zip(player_names, shap_values, strict=True):
                row[f"decision_kernel_shap_{sample_count}_{player_name}"] = float(
                    shap_value
                )
        for player_name, activation_rate, activated_value_sum, activated_value in zip(
            player_names,
            decision_activation.activation_rate,
            decision_activation.activated_value_sum,
            decision_activation.activated_value,
            strict=True,
        ):
            row[f"decision_activation_rate_{player_name}"] = float(activation_rate)
            row[f"decision_activated_value_sum_{player_name}"] = float(
                activated_value_sum
            )
            row[f"decision_activated_value_{player_name}"] = float(activated_value)
        if ante_decision_shap is not None:
            if (
                ante_decision_values is None
                or ante_decision_characteristic_values is None
            ):
                raise RuntimeError("Ante InfoDVA values were not computed.")
            row["ante_decision_baseline_value"] = float(ante_decision_values[0])
            row["ante_decision_full_value"] = float(ante_decision_values[-1])
            row["ante_decision_value_gain"] = float(
                ante_decision_characteristic_values[-1]
            )
            for player_name, shap_value in zip(
                player_names,
                ante_decision_shap,
                strict=True,
            ):
                row[f"ante_decision_shap_{player_name}"] = float(shap_value)
        if cvar_decision_shap is not None:
            if (
                cvar_decision_values is None
                or cvar_decision_characteristic_values is None
                or cvar_baseline_solution is None
                or cvar_full_solution is None
                or cvar_decision_activation is None
            ):
                raise RuntimeError("CVaR SHAP values require CVaR coverage solutions.")
            row["cvar_alpha"] = float(config.cvar_alpha)
            row["cvar_scenario_count"] = int(config.cvar_scenario_count)
            row["cvar_decision_baseline_value"] = float(cvar_decision_values[0])
            row["cvar_decision_full_value"] = float(cvar_decision_values[-1])
            row["cvar_decision_value_gain"] = float(
                cvar_decision_characteristic_values[-1]
            )
            row["cvar_actual_regret"] = float(oracle_value - cvar_decision_values[-1])
            row["cvar_decision_baseline_covered_demand"] = _covered_demand_value(
                cvar_baseline_solution.covered_zone_indices,
                true_demand,
            )
            row["cvar_decision_full_covered_demand"] = _covered_demand_value(
                cvar_full_solution.covered_zone_indices,
                true_demand,
            )
            row["cvar_baseline_selected_zip_codes"] = json.dumps(
                list(cvar_baseline_solution.selected_facility_zip_codes)
            )
            row["cvar_full_selected_zip_codes"] = json.dumps(
                list(cvar_full_solution.selected_facility_zip_codes)
            )
            row["cvar_baseline_covered_zip_codes"] = json.dumps(
                list(cvar_baseline_solution.covered_zip_codes)
            )
            row["cvar_full_covered_zip_codes"] = json.dumps(
                list(cvar_full_solution.covered_zip_codes)
            )
            row["cvar_baseline_risk_objective_value"] = (
                None
                if cvar_baseline_solution.risk_objective_value is None
                else float(cvar_baseline_solution.risk_objective_value)
            )
            row["cvar_full_risk_objective_value"] = (
                None
                if cvar_full_solution.risk_objective_value is None
                else float(cvar_full_solution.risk_objective_value)
            )
            for player_name, shap_value in zip(
                player_names,
                cvar_decision_shap,
                strict=True,
            ):
                row[f"cvar_decision_shap_{player_name}"] = float(shap_value)
            for (
                player_name,
                activation_rate,
                activated_value_sum,
                activated_value,
            ) in zip(
                player_names,
                cvar_decision_activation.activation_rate,
                cvar_decision_activation.activated_value_sum,
                cvar_decision_activation.activated_value,
                strict=True,
            ):
                row[f"cvar_decision_activation_rate_{player_name}"] = float(
                    activation_rate
                )
                row[f"cvar_decision_activated_value_sum_{player_name}"] = float(
                    activated_value_sum
                )
                row[f"cvar_decision_activated_value_{player_name}"] = float(
                    activated_value
                )
        hourly_rows.append(row)

        for player_idx, player_name in enumerate(player_names):
            for output_idx, shap_value in enumerate(predictive_zip_shap[player_idx]):
                predictive_zip_rows.append(
                    {
                        "timestamp_hour": timestamp_key,
                        "player": player_name,
                        "zip_code": zip_codes[output_idx],
                        "target_column": target_columns[output_idx],
                        "output_index": output_idx,
                        "shap_value": float(shap_value),
                    }
                )

        coverage_solution_rows.extend(
            [
                _build_coverage_solution_row(
                    timestamp_key,
                    "baseline_model",
                    baseline_solution,
                    true_demand,
                    config,
                ),
                _build_coverage_solution_row(
                    timestamp_key,
                    "full_model",
                    full_solution,
                    true_demand,
                    config,
                ),
                _build_coverage_solution_row(
                    timestamp_key,
                    "oracle_truth",
                    oracle_solution,
                    true_demand,
                    config,
                ),
            ]
        )
        if config.compute_cvar_decision_shap:
            if cvar_baseline_solution is None or cvar_full_solution is None:
                raise RuntimeError("Expected CVaR coverage solutions when enabled.")
            coverage_solution_rows.extend(
                [
                    _build_coverage_solution_row(
                        timestamp_key,
                        "cvar_baseline_model",
                        cvar_baseline_solution,
                        true_demand,
                        config,
                    ),
                    _build_coverage_solution_row(
                        timestamp_key,
                        "cvar_full_model",
                        cvar_full_solution,
                        true_demand,
                        config,
                    ),
                ]
            )
        if config.save_coalition_values:
            coalition_rows.extend(
                _build_coalition_rows(
                    timestamp_key,
                    predictive_values,
                    decision_values,
                    decision_solutions,
                    player_names,
                    ante_decision_values=ante_decision_values,
                    cvar_decision_values=cvar_decision_values,
                    cvar_decision_solutions=cvar_decision_solutions,
                )
            )
        print(
            f"[{hour_idx}/{len(explain_timestamps)}] explained {timestamp_key} "
            f"in {time.perf_counter() - hour_started_at:.2f}s",
            flush=True,
        )

    hourly_shap = pd.DataFrame(hourly_rows)
    summary_shap = _build_summary_shap_frame(hourly_shap, player_names)
    cvar_summary_shap = (
        _build_cvar_summary_shap_frame(hourly_shap, player_names)
        if config.compute_cvar_decision_shap
        else None
    )
    evaluation_metrics = {
        metric_name: build_metric_summary(values_by_hour)
        for metric_name, values_by_hour in metric_values.items()
    }
    run_metadata = {
        "x_path": str(config.x_path),
        "y_path": str(config.y_path),
        "metadata_path": None if config.metadata_path is None else str(config.metadata_path),
        "zone_order_path": str(config.zone_order_path),
        "distance_matrix_path": str(config.distance_matrix_path),
        "feature_columns": list(feature_columns),
        "feature_groups": {
            group.name: list(group.columns)
            for group in feature_groups
        },
        "player_names": list(player_names),
        "player_count": len(player_names),
        "target_columns": list(target_columns),
        "zip_codes": list(zip_codes),
        "coverage_radius_km": float(config.coverage_radius_km),
        "facility_budget": int(config.facility_budget),
        "coverage_solver": coverage_solver,
        "coverage_backend_solver": normalize_optimization_solver(config.optimization_solver),
        "oracle_solver": EMS_COVERAGE_SOLVER_EXACT,
        "oracle_backend_solver": normalize_optimization_solver(config.optimization_solver),
        "coverage_matrix_density": float(np.mean(coverage_matrix)),
        "model_name": "XGBRegressor",
        "model_id": str(config.model_id),
        "xgb_params": _build_xgb_params(config),
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
        "train_rows": int(len(train_x)),
        "candidate_train_rows": int(unsampled_train_rows),
        "train_sample_rows": config.train_sample_rows,
        "train_start": str(time_split.train_start),
        "train_end": str(time_split.train_end),
        "holdout_rows": int(len(holdout_x)),
        "test_months": int(config.test_months),
        "holdout_start": str(time_split.holdout_start),
        "holdout_end": str(time_split.holdout_end),
        "holdout_hours_requested": int(config.holdout_hours),
        "explained_hour_sample_size": int(len(explain_x)),
        "explanation_sample_seed": int(config.random_state),
        "explained_row_index_basis": (
            "0-based row positions after sorting X/y tables by timestamp_hour"
        ),
        "explained_rows": list(explained_rows),
        "explained_hours": [str(timestamp) for timestamp in explain_timestamps],
        "holdout_hours": int(config.holdout_hours),
        "max_hours": config.max_hours,
        "background_rows": int(config.background_rows),
        "coalition_count": 1 << len(player_names),
        "coalition_batch_size": int(config.coalition_batch_size),
        "progress_every_coalitions": int(config.progress_every_coalitions),
        "solver_params": _build_solver_params(config),
        "objective_tolerance": float(config.objective_tolerance),
        "decision_activation_metric_enabled": True,
        "decision_activation_definition": EMS_DECISION_ACTIVATION_DEFINITION,
        "compute_ante_decision_shap": bool(config.compute_ante_decision_shap),
        "ante_decision_value_definition": (
            "J(yhat_full, w(S)) - J(yhat_full, w(empty))"
        ),
        "decision_permutation_shap_samples": list(decision_permutation_sample_counts),
        "decision_kernel_shap_samples": list(decision_kernel_sample_counts),
        "decision_permutation_shap_seed": int(decision_permutation_seed),
        "decision_kernel_shap_seed": int(decision_kernel_seed),
        "cvar_decision_activation_metric_enabled": bool(
            config.compute_cvar_decision_shap
        ),
        "compute_cvar_decision_shap": bool(config.compute_cvar_decision_shap),
        "cvar_alpha": float(config.cvar_alpha),
        "cvar_scenario_count": int(config.cvar_scenario_count),
        "cvar_scenario_method": "training_residual_bootstrap",
        "cvar_predictive_model_changed": False,
        "shap_method": "exact_grouped_coalition_enumeration_empirical_background",
        "runtime_seconds": time.perf_counter() - started_at,
    }

    coalition_values = pd.DataFrame(coalition_rows) if config.save_coalition_values else None
    return EmsExactShapOutputs(
        hourly_shap=hourly_shap,
        predictive_zip_shap=pd.DataFrame(predictive_zip_rows),
        coverage_solutions=pd.DataFrame(coverage_solution_rows),
        summary_shap=summary_shap,
        prediction_metrics=prediction_metrics,
        evaluation_metrics=evaluation_metrics,
        run_metadata=run_metadata,
        coalition_values=coalition_values,
        cvar_summary_shap=cvar_summary_shap,
    )


def write_ems_exact_shap_outputs(
    outputs: EmsExactShapOutputs,
    outdir: Path | str,
) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    outputs.hourly_shap.to_csv(outdir_path / "hourly_shap.csv", index=False)
    outputs.predictive_zip_shap.to_csv(
        outdir_path / "predictive_zip_shap.csv",
        index=False,
    )
    outputs.coverage_solutions.to_csv(
        outdir_path / "coverage_solutions.csv",
        index=False,
    )
    outputs.summary_shap.to_csv(outdir_path / "summary_shap.csv", index=False)
    cvar_summary_path = outdir_path / "cvar_summary_shap.csv"
    if outputs.cvar_summary_shap is not None:
        outputs.cvar_summary_shap.to_csv(cvar_summary_path, index=False)
    else:
        cvar_summary_path.unlink(missing_ok=True)
    coalition_values_path = outdir_path / "coalition_values.csv"
    if outputs.coalition_values is not None:
        outputs.coalition_values.to_csv(coalition_values_path, index=False)
    else:
        coalition_values_path.unlink(missing_ok=True)
    with (outdir_path / "prediction_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.prediction_metrics, handle, indent=2, sort_keys=True)
    with (outdir_path / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.evaluation_metrics, handle, indent=2, sort_keys=True)
    with (outdir_path / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs.run_metadata, handle, indent=2, sort_keys=True)


def load_ems_exact_shap_outputs(results_dir: Path | str) -> EmsExactShapOutputs:
    results_dir_path = Path(results_dir)
    hourly_shap = pd.read_csv(results_dir_path / "hourly_shap.csv")
    predictive_zip_shap = pd.read_csv(results_dir_path / "predictive_zip_shap.csv")
    coverage_solutions = pd.read_csv(results_dir_path / "coverage_solutions.csv")
    summary_shap = pd.read_csv(results_dir_path / "summary_shap.csv")
    cvar_summary_path = results_dir_path / "cvar_summary_shap.csv"
    cvar_summary_shap = (
        pd.read_csv(cvar_summary_path)
        if cvar_summary_path.exists()
        else None
    )
    coalition_values_path = results_dir_path / "coalition_values.csv"
    coalition_values = (
        pd.read_csv(coalition_values_path)
        if coalition_values_path.exists()
        else None
    )
    with (results_dir_path / "prediction_metrics.json").open(encoding="utf-8") as handle:
        prediction_metrics = json.load(handle)
    with (results_dir_path / "evaluation_metrics.json").open(encoding="utf-8") as handle:
        evaluation_metrics = json.load(handle)
    with (results_dir_path / "run_metadata.json").open(encoding="utf-8") as handle:
        run_metadata = json.load(handle)
    return EmsExactShapOutputs(
        hourly_shap=hourly_shap,
        predictive_zip_shap=predictive_zip_shap,
        coverage_solutions=coverage_solutions,
        summary_shap=summary_shap,
        prediction_metrics=prediction_metrics,
        evaluation_metrics=evaluation_metrics,
        run_metadata=run_metadata,
        coalition_values=coalition_values,
        cvar_summary_shap=cvar_summary_shap,
    )


def _validate_config(config: EmsExactShapConfig) -> None:
    if config.holdout_hours <= 0:
        raise ValueError("holdout_hours must be strictly positive.")
    if config.test_months <= 0:
        raise ValueError("test_months must be strictly positive.")
    if config.max_hours is not None and config.max_hours <= 0:
        raise ValueError("max_hours must be strictly positive when provided.")
    if config.background_rows <= 0:
        raise ValueError("background_rows must be strictly positive.")
    if config.coalition_batch_size <= 0:
        raise ValueError("coalition_batch_size must be strictly positive.")
    if config.progress_every_coalitions < 0:
        raise ValueError("progress_every_coalitions must be non-negative.")
    if config.xgb_n_estimators <= 0:
        raise ValueError("xgb_n_estimators must be strictly positive.")
    if config.xgb_max_depth <= 0:
        raise ValueError("xgb_max_depth must be strictly positive.")
    if config.train_sample_rows is not None and config.train_sample_rows <= 0:
        raise ValueError("train_sample_rows must be strictly positive when provided.")
    if config.coverage_radius_km < 0:
        raise ValueError("coverage_radius_km must be non-negative.")
    if config.facility_budget < 0:
        raise ValueError("facility_budget must be non-negative.")
    if config.gurobi_threads <= 0:
        raise ValueError("gurobi_threads/solver_threads must be strictly positive.")
    if config.objective_tolerance < 0:
        raise ValueError("objective_tolerance must be non-negative.")
    if not 0.0 <= config.cvar_alpha < 1.0:
        raise ValueError("cvar_alpha must be in [0, 1).")
    if config.cvar_scenario_count <= 0:
        raise ValueError("cvar_scenario_count must be strictly positive.")
    _normalize_decision_shap_sample_counts(
        config.decision_permutation_shap_samples,
        "decision_permutation_shap_samples",
    )
    _normalize_decision_shap_sample_counts(
        config.decision_kernel_shap_samples,
        "decision_kernel_shap_samples",
    )
    _resolve_decision_shap_seed(
        config.decision_permutation_shap_seed,
        config.random_state,
        "decision_permutation_shap_seed",
    )
    _resolve_decision_shap_seed(
        config.decision_kernel_shap_seed,
        config.random_state,
        "decision_kernel_shap_seed",
    )
    normalize_ems_coverage_solver(config.coverage_solver)
    normalize_optimization_solver(config.optimization_solver)


def _load_ems_frames(
    config: EmsExactShapConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not config.x_path.exists():
        raise FileNotFoundError(f"Missing EMS X table: {config.x_path}")
    if not config.y_path.exists():
        raise FileNotFoundError(f"Missing EMS y table: {config.y_path}")

    x_frame = pd.read_csv(config.x_path, parse_dates=[EMS_TIMESTAMP_COLUMN])
    y_frame = pd.read_csv(config.y_path, parse_dates=[EMS_TIMESTAMP_COLUMN])
    metadata = _load_metadata(config.metadata_path)
    if EMS_TIMESTAMP_COLUMN not in x_frame.columns or EMS_TIMESTAMP_COLUMN not in y_frame.columns:
        raise KeyError(f"Both EMS X and y tables must include {EMS_TIMESTAMP_COLUMN}.")
    _validate_excluded_zip_codes_absent(config, x_frame, y_frame, metadata)

    x_frame = x_frame.sort_values(EMS_TIMESTAMP_COLUMN).reset_index(drop=True)
    y_frame = y_frame.sort_values(EMS_TIMESTAMP_COLUMN).reset_index(drop=True)
    if not x_frame[EMS_TIMESTAMP_COLUMN].equals(y_frame[EMS_TIMESTAMP_COLUMN]):
        raise ValueError("EMS X and y timestamp columns are not aligned.")
    return x_frame, y_frame, metadata


def _load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("EMS metadata JSON must contain an object.")
    return metadata


def _validate_excluded_zip_codes_absent(
    config: EmsExactShapConfig,
    x_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> None:
    excluded_zip_codes = {str(zip_code) for zip_code in config.excluded_zip_codes}
    if not excluded_zip_codes:
        return

    offending_columns = sorted(
        column_name
        for column_name in [*x_frame.columns, *y_frame.columns]
        for zip_code in excluded_zip_codes
        if str(column_name).endswith(f"_zip_{zip_code}")
    )
    metadata_zip_codes = metadata.get("zip_codes")
    offending_metadata_zips = (
        sorted(set(map(str, metadata_zip_codes)) & excluded_zip_codes)
        if isinstance(metadata_zip_codes, list)
        else []
    )
    if not offending_columns and not offending_metadata_zips:
        return

    details = []
    if offending_metadata_zips:
        details.append("metadata zip_codes=" + ", ".join(offending_metadata_zips))
    if offending_columns:
        details.append("columns=" + ", ".join(offending_columns[:8]))
        if len(offending_columns) > 8:
            details[-1] += f", ... ({len(offending_columns)} total)"
    raise ValueError(
        "EMS inputs still contain excluded ZIP codes. Regenerate the EMS processed "
        "tables before running SHAP: " + "; ".join(details)
    )


def _load_zone_order(
    path: Path,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing EMS ZIP order table: {path}")

    zone_order = pd.read_csv(path, dtype={"zip_code": str, "modzcta": str})
    if "output_index" in zone_order.columns:
        zone_order = zone_order.sort_values("output_index")
    required_columns = {"zip_code", "target_column"}
    missing_columns = required_columns.difference(zone_order.columns)
    if missing_columns:
        raise ValueError(
            "Zone order table is missing columns: " + ", ".join(sorted(missing_columns))
        )
    zone_order["target_column"] = zone_order["target_column"].astype(str)
    missing_targets = sorted(set(zone_order["target_column"]) - set(target_columns))
    if missing_targets:
        raise ValueError(
            "EMS y table is missing zone-order target columns: "
            + ", ".join(missing_targets)
        )
    return zone_order.reset_index(drop=True)


def _resolve_feature_columns(
    x_frame: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    raw_feature_columns = metadata.get("feature_columns")
    if isinstance(raw_feature_columns, list) and raw_feature_columns:
        feature_columns = tuple(str(column) for column in raw_feature_columns)
    else:
        feature_columns = tuple(
            column for column in x_frame.columns if column != EMS_TIMESTAMP_COLUMN
        )
    missing_columns = sorted(set(feature_columns) - set(x_frame.columns))
    if missing_columns:
        raise ValueError(
            "EMS X table is missing metadata feature columns: "
            + ", ".join(missing_columns)
        )
    return tuple(column for column in feature_columns if column != MONTH_FEATURE_COLUMN)


def _target_columns(y_frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in y_frame.columns if column != EMS_TIMESTAMP_COLUMN)


def _build_time_split(
    x_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    *,
    test_months: int,
) -> EmsTimeSplit:
    unique_hours = pd.DatetimeIndex(
        x_frame[EMS_TIMESTAMP_COLUMN].drop_duplicates(),
    ).sort_values()
    if unique_hours.empty:
        raise ValueError("EMS X table does not contain any timestamps.")
    latest_hour = cast(pd.Timestamp, pd.Timestamp(unique_hours[-1]))
    holdout_period = cast(
        pd.Period,
        latest_hour.to_period("M") - (int(test_months) - 1),
    )
    holdout_start = holdout_period.to_timestamp()
    is_holdout = x_frame[EMS_TIMESTAMP_COLUMN] >= holdout_start
    if not is_holdout.any():
        raise ValueError(
            f"No holdout rows found for final {test_months} calendar month(s)."
        )
    if bool(is_holdout.all()):
        raise ValueError(
            f"Final {test_months} calendar month(s) contain all "
            f"{len(unique_hours)} available hours; no training rows remain."
        )
    source_rows = np.arange(len(x_frame), dtype=int)
    holdout_mask = is_holdout.to_numpy(dtype=bool, copy=True)
    train_x = x_frame.loc[~is_holdout].reset_index(drop=True)
    train_y = y_frame.loc[~is_holdout].reset_index(drop=True)
    holdout_x = x_frame.loc[is_holdout].reset_index(drop=True)
    holdout_y = y_frame.loc[is_holdout].reset_index(drop=True)
    return EmsTimeSplit(
        train_x=train_x,
        train_y=train_y,
        holdout_x=holdout_x,
        holdout_y=holdout_y,
        train_source_rows=tuple(int(row) for row in source_rows[~holdout_mask]),
        holdout_source_rows=tuple(int(row) for row in source_rows[holdout_mask]),
        train_start=cast(pd.Timestamp, pd.Timestamp(train_x[EMS_TIMESTAMP_COLUMN].iloc[0])),
        train_end=cast(pd.Timestamp, pd.Timestamp(train_x[EMS_TIMESTAMP_COLUMN].iloc[-1])),
        holdout_start=cast(
            pd.Timestamp,
            pd.Timestamp(holdout_x[EMS_TIMESTAMP_COLUMN].iloc[0]),
        ),
        holdout_end=cast(
            pd.Timestamp,
            pd.Timestamp(holdout_x[EMS_TIMESTAMP_COLUMN].iloc[-1]),
        ),
    )


def _sample_explanation_hours(
    holdout_x: pd.DataFrame,
    holdout_y: pd.DataFrame,
    holdout_source_rows: Sequence[int],
    *,
    holdout_hours: int,
    max_hours: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[dict[str, Any], ...]]:
    sample_size = holdout_hours if max_hours is None else min(holdout_hours, max_hours)
    if sample_size > len(holdout_x):
        raise ValueError(
            f"Requested {sample_size} explanation hours but the holdout/test set "
            f"contains only {len(holdout_x)} hours."
        )
    if len(holdout_source_rows) != len(holdout_x):
        raise ValueError("holdout_source_rows must align with holdout_x.")

    rng = np.random.default_rng(random_state)
    sampled_positions = np.sort(
        rng.choice(len(holdout_x), size=sample_size, replace=False),
    )
    explain_x = holdout_x.iloc[sampled_positions].reset_index(drop=True)
    explain_y = holdout_y.iloc[sampled_positions].reset_index(drop=True)
    explained_rows = tuple(
        {
            "explanation_index": int(explanation_idx),
            "holdout_position": int(holdout_position),
            "source_row_position": int(holdout_source_rows[int(holdout_position)]),
            "timestamp_hour": str(
                holdout_x.iloc[int(holdout_position)][EMS_TIMESTAMP_COLUMN]
            ),
        }
        for explanation_idx, holdout_position in enumerate(sampled_positions)
    )
    return explain_x, explain_y, explained_rows


def _maybe_sample_training_rows(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    *,
    train_sample_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_sample_rows is None or train_sample_rows >= len(train_x):
        return train_x, train_y

    sampled_positions = (
        train_x.sample(n=train_sample_rows, random_state=random_state)
        .sort_values(EMS_TIMESTAMP_COLUMN)
        .index
    )
    return (
        train_x.loc[sampled_positions].reset_index(drop=True),
        train_y.loc[sampled_positions].reset_index(drop=True),
    )


def _fit_xgb_regressor(
    *,
    train_frame: pd.DataFrame,
    y_train: pd.DataFrame,
    feature_columns: Sequence[str],
    config: EmsExactShapConfig,
) -> XGBRegressor:
    model = XGBRegressor(**_build_xgb_params(config))
    model.fit(
        _frame_to_feature_matrix(train_frame, feature_columns),
        _as_float32_matrix(y_train.to_numpy(dtype=np.float32, copy=True)),
    )
    return model


def _build_xgb_params(config: EmsExactShapConfig) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": config.xgb_n_estimators,
        "max_depth": config.xgb_max_depth,
        "learning_rate": config.xgb_learning_rate,
        "subsample": config.xgb_subsample,
        "colsample_bytree": config.xgb_colsample_bytree,
        "reg_lambda": config.xgb_reg_lambda,
        "tree_method": "hist",
        "random_state": config.random_state,
        "n_jobs": 1,
        "verbosity": config.xgb_verbosity,
    }


def _build_prediction_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must share the same shape for metric computation. "
            f"Got {y_true.shape} and {y_pred.shape}."
        )

    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "holdout": {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "hours": int(y_true.shape[0]),
            "targets_per_hour": int(y_true.shape[1]) if y_true.ndim > 1 else 1,
            "predictions": int(y_true.size),
        }
    }


def _build_training_residual_matrix(
    *,
    model: Any,
    train_frame: pd.DataFrame,
    train_y: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
) -> np.ndarray:
    train_predictions = np.maximum(
        np.asarray(
            model.predict(_frame_to_feature_matrix(train_frame, feature_columns)),
            dtype=float,
        ),
        0.0,
    )
    if train_predictions.ndim == 1:
        train_predictions = train_predictions[:, np.newaxis]

    observed = train_y.loc[:, list(target_columns)].to_numpy(dtype=float, copy=True)
    if observed.shape != train_predictions.shape:
        raise ValueError(
            "Training observations and predictions must have the same shape for "
            f"residuals. Got {observed.shape} and {train_predictions.shape}."
        )
    return observed - train_predictions


def _sample_residual_scenarios(
    residual_matrix: np.ndarray,
    *,
    scenario_count: int,
    random_state: int,
) -> np.ndarray:
    residual_array = np.asarray(residual_matrix, dtype=float)
    if residual_array.ndim != 2:
        raise ValueError("residual_matrix must have shape (rows, targets).")
    if len(residual_array) <= 0:
        raise ValueError("Need at least one residual row.")
    if scenario_count <= 0:
        raise ValueError("scenario_count must be strictly positive.")

    rng = np.random.default_rng(random_state)
    sampled_rows = rng.choice(
        len(residual_array),
        size=int(scenario_count),
        replace=True,
    )
    return residual_array[sampled_rows]


def _build_coalition_demand_scenarios(
    coalition_predictions: np.ndarray,
    residual_scenarios: np.ndarray,
) -> np.ndarray:
    point_predictions = np.asarray(coalition_predictions, dtype=float)
    residual_array = np.asarray(residual_scenarios, dtype=float)
    if point_predictions.ndim != 2:
        raise ValueError(
            "coalition_predictions must have shape (coalition_count, target_count)."
        )
    if residual_array.ndim != 2:
        raise ValueError(
            "residual_scenarios must have shape (scenario_count, target_count)."
        )
    if point_predictions.shape[1] != residual_array.shape[1]:
        raise ValueError(
            "coalition_predictions and residual_scenarios must have the same "
            f"target_count. Got {point_predictions.shape[1]} and "
            f"{residual_array.shape[1]}."
        )

    return np.maximum(
        point_predictions[:, np.newaxis, :] + residual_array[np.newaxis, :, :],
        0.0,
    )


def _frame_to_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    return _as_float32_matrix(frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float32))


def _as_float32_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    return np.ascontiguousarray(array)


def _sample_background_frame(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    rows: int,
    random_state: int,
) -> pd.DataFrame:
    if rows > len(train_frame):
        raise ValueError(
            f"Requested {rows} background rows but only "
            f"{len(train_frame)} training rows are available."
        )
    return (
        train_frame.loc[:, list(feature_columns)]
        .sample(n=rows, random_state=random_state)
        .reset_index(drop=True)
    )


def _load_distance_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing EMS distance matrix: {path}")
    if path.suffix == ".parquet":
        distance_matrix = pd.read_parquet(path)
        if "zip_code" in distance_matrix.columns:
            distance_matrix = distance_matrix.set_index("zip_code")
    elif path.suffix == ".csv":
        distance_matrix = pd.read_csv(path, dtype={"zip_code": str})
        if "zip_code" not in distance_matrix.columns:
            raise ValueError("CSV distance matrix must include a zip_code column.")
        distance_matrix = distance_matrix.set_index("zip_code")
    else:
        raise ValueError("Distance matrix must be a parquet or CSV file.")
    return _normalize_distance_matrix(distance_matrix)


def _normalize_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    normalized = distance_matrix.copy()
    normalized.index = normalized.index.astype(str)
    normalized.columns = [str(column) for column in normalized.columns]
    if normalized.isna().any().any():
        raise ValueError("Distance matrix contains missing values.")
    return normalized.astype(float).sort_index().sort_index(axis=1)


def _solve_decision_values(
    *,
    coalition_demand_matrix: np.ndarray,
    true_demand: np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    config: EmsExactShapConfig,
    coverage_solver: str | None = None,
    progress_label: str | None = None,
) -> tuple[
    np.ndarray,
    MaximumCoverageResult,
    MaximumCoverageResult,
    tuple[MaximumCoverageResult, ...],
]:
    decision_values = np.empty(coalition_demand_matrix.shape[0], dtype=float)
    coalition_solutions: list[MaximumCoverageResult] = []
    baseline_solution = None
    full_solution = None
    full_mask = coalition_demand_matrix.shape[0] - 1
    solver_name = (
        normalize_ems_coverage_solver(coverage_solver)
        if coverage_solver is not None
        else normalize_ems_coverage_solver(config.coverage_solver)
    )
    for coalition_mask, predicted_demand in enumerate(coalition_demand_matrix):
        solution = solve_ems_coverage(
            predicted_demand,
            coverage_matrix,
            zip_codes,
            facility_budget=config.facility_budget,
            solver_name=solver_name,
            name=f"ems_max_coverage_{coalition_mask}",
            solver_params=_build_solver_params(config),
            optimization_solver=config.optimization_solver,
            objective_tolerance=config.objective_tolerance,
        )
        decision_values[coalition_mask] = _realized_coverage_value(
            solution.covered_zone_indices,
            true_demand,
        )
        coalition_solutions.append(solution)
        if coalition_mask == 0:
            baseline_solution = solution
        if coalition_mask == full_mask:
            full_solution = solution
        completed_count = coalition_mask + 1
        if (
            progress_label is not None
            and config.progress_every_coalitions > 0
            and (
                completed_count == len(coalition_demand_matrix)
                or completed_count % config.progress_every_coalitions == 0
            )
        ):
            print(
                f"{progress_label}: solved {completed_count:,}/"
                f"{len(coalition_demand_matrix):,} max-coverage models",
                flush=True,
            )
    if baseline_solution is None or full_solution is None:
        raise RuntimeError("Expected baseline and full-coalition coverage solutions.")
    return decision_values, baseline_solution, full_solution, tuple(coalition_solutions)


def _solve_cvar_decision_values(
    *,
    coalition_demand_scenarios: np.ndarray,
    true_demand: np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    config: EmsExactShapConfig,
    progress_label: str | None = None,
) -> tuple[
    np.ndarray,
    MaximumCoverageResult,
    MaximumCoverageResult,
    tuple[MaximumCoverageResult, ...],
]:
    scenario_array = np.asarray(coalition_demand_scenarios, dtype=float)
    if scenario_array.ndim != 3:
        raise ValueError(
            "coalition_demand_scenarios must have shape "
            "(coalition_count, scenario_count, demand_count)."
        )

    decision_values = np.empty(scenario_array.shape[0], dtype=float)
    coalition_solutions: list[MaximumCoverageResult] = []
    baseline_solution = None
    full_solution = None
    full_mask = scenario_array.shape[0] - 1
    for coalition_mask, demand_scenarios in enumerate(scenario_array):
        solution = solve_cvar_coverage(
            demand_scenarios,
            coverage_matrix,
            zip_codes,
            facility_budget=config.facility_budget,
            alpha=config.cvar_alpha,
            name=f"ems_cvar_coverage_{coalition_mask}",
            solver_params=_build_solver_params(config),
            optimization_solver=config.optimization_solver,
            objective_tolerance=config.objective_tolerance,
        )
        decision_values[coalition_mask] = _realized_coverage_value(
            solution.covered_zone_indices,
            true_demand,
        )
        coalition_solutions.append(solution)
        if coalition_mask == 0:
            baseline_solution = solution
        if coalition_mask == full_mask:
            full_solution = solution
        completed_count = coalition_mask + 1
        if (
            progress_label is not None
            and config.progress_every_coalitions > 0
            and (
                completed_count == len(scenario_array)
                or completed_count % config.progress_every_coalitions == 0
            )
        ):
            print(
                f"{progress_label}: solved {completed_count:,}/"
                f"{len(scenario_array):,} CVaR coverage models",
                flush=True,
            )
    if baseline_solution is None or full_solution is None:
        raise RuntimeError("Expected baseline and full-coalition CVaR coverage solutions.")
    return decision_values, baseline_solution, full_solution, tuple(coalition_solutions)


def _prepare_coverage_inputs(
    demand: Sequence[float] | np.ndarray,
    coverage_matrix: np.ndarray,
    zip_codes: Sequence[str],
    *,
    facility_budget: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    demand_array = np.maximum(np.asarray(demand, dtype=float), 0.0)
    coverage_array = np.asarray(coverage_matrix, dtype=bool)
    ordered_zip_codes = tuple(str(zip_code) for zip_code in zip_codes)
    if demand_array.ndim != 1:
        raise ValueError("demand must be a one-dimensional vector.")
    if coverage_array.shape != (len(demand_array), len(ordered_zip_codes)):
        raise ValueError(
            "coverage_matrix must have shape (demand_count, facility_count). "
            f"Got {coverage_array.shape}, expected "
            f"{(len(demand_array), len(ordered_zip_codes))}."
        )
    if facility_budget < 0:
        raise ValueError("facility_budget must be non-negative.")
    return demand_array, coverage_array, ordered_zip_codes


def _build_maximum_coverage_pyomo_model(
    *,
    name: str,
    demand_array: np.ndarray,
    coverage_array: np.ndarray,
    facility_budget: int,
    relax_integrality: bool,
) -> pyo.ConcreteModel:
    demand_count, facility_count = coverage_array.shape
    demand_total = _total_demand(demand_array)
    variable_domain = pyo.UnitInterval if relax_integrality else pyo.Binary

    model = pyo.ConcreteModel(name=name)
    model.FACILITIES = pyo.Set(initialize=range(facility_count))
    model.DEMANDS = pyo.Set(initialize=range(demand_count))
    model.x = pyo.Var(model.FACILITIES, domain=variable_domain)
    model.y = pyo.Var(model.DEMANDS, domain=variable_domain)

    def coverage_rule(coverage_model: pyo.ConcreteModel, demand_idx: int) -> pyo.Expression:
        covering_facilities = [
            facility_idx
            for facility_idx in range(facility_count)
            if coverage_array[demand_idx, facility_idx]
        ]
        return coverage_model.y[demand_idx] <= sum(
            coverage_model.x[facility_idx] for facility_idx in covering_facilities
        )

    model.coverage = pyo.Constraint(model.DEMANDS, rule=coverage_rule)
    model.facility_budget = pyo.Constraint(
        expr=sum(model.x[facility_idx] for facility_idx in range(facility_count))
        <= int(facility_budget)
    )
    model.primary_objective_expression = pyo.Expression(
        expr=sum(
            _coverage_weight(float(demand_array[demand_idx]), demand_total)
            * model.y[demand_idx]
            for demand_idx in range(demand_count)
        )
    )
    model.objective = pyo.Objective(
        expr=model.primary_objective_expression,
        sense=pyo.maximize,
    )
    return model


def _solve_coverage_model(
    model: pyo.ConcreteModel,
    *,
    label: str,
    solver_params: Mapping[str, float | int | str] | None,
    optimization_solver: str,
    log_to_console: bool,
):
    effective_solver_params = dict(solver_params or {})
    effective_solver_params.setdefault("FeasibilityTol", 1e-9)
    solve_result = solve_pyomo_model(
        model,
        solver_name=optimization_solver,
        solver_params=effective_solver_params,
        log_to_console=log_to_console,
    )
    require_pyomo_solution(solve_result, problem_label=label)
    return solve_result


def _validate_zip_aligned_demand(
    demand_array: np.ndarray,
    ordered_zip_codes: Sequence[str],
    solver_name: str,
) -> None:
    if len(demand_array) != len(ordered_zip_codes):
        raise ValueError(
            f"{solver_name} requires one demand value per candidate ZIP code. "
            f"Got {len(demand_array)} demand values and "
            f"{len(ordered_zip_codes)} ZIP codes."
        )


def _rank_indices_descending(values: np.ndarray) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(values)),
            key=lambda value_idx: (-float(values[value_idx]), value_idx),
        )
    )


def _round_lp_relaxation_facility_indices(
    relaxed_facility_values: np.ndarray,
    *,
    facility_budget: int,
) -> tuple[int, ...]:
    max_facilities = min(max(0, int(facility_budget)), len(relaxed_facility_values))
    ranked_indices = _rank_indices_descending(relaxed_facility_values)
    selected = [
        facility_idx
        for facility_idx in ranked_indices[:max_facilities]
        if float(relaxed_facility_values[facility_idx]) > _LP_RELAXATION_ROUNDING_TOLERANCE
    ]
    return tuple(sorted(selected))


def _build_coverage_result(
    *,
    demand_array: np.ndarray,
    coverage_array: np.ndarray,
    ordered_zip_codes: Sequence[str],
    selected_indices: Sequence[int],
    solver_status: int | str,
    optimal: bool,
    mip_gap: float | None,
    solver_name: str,
    optimization_solver: str = DEFAULT_OPTIMIZATION_SOLVER,
    solver_runtime_seconds: float | None = None,
    risk_objective_value: float | None = None,
) -> MaximumCoverageResult:
    selected_index_set = sorted({int(facility_idx) for facility_idx in selected_indices})
    facility_count = coverage_array.shape[1]
    invalid_indices = [
        facility_idx
        for facility_idx in selected_index_set
        if facility_idx < 0 or facility_idx >= facility_count
    ]
    if invalid_indices:
        raise ValueError(
            "selected_indices contains invalid facility indices: "
            + ", ".join(str(index) for index in invalid_indices)
        )

    if selected_index_set:
        covered_mask = np.asarray(
            coverage_array[:, selected_index_set].any(axis=1),
            dtype=bool,
        ).reshape(len(demand_array))
    else:
        covered_mask = np.zeros(len(demand_array), dtype=bool)
    covered_indices = tuple(
        demand_idx for demand_idx, is_covered in enumerate(covered_mask) if is_covered
    )
    selected_indices_tuple = tuple(selected_index_set)
    covered_demand = _covered_demand_value(covered_indices, demand_array)
    total_demand = _total_demand(demand_array)
    return MaximumCoverageResult(
        objective_value=_coverage_fraction(covered_demand, total_demand),
        covered_demand=covered_demand,
        total_demand=total_demand,
        selected_facility_zip_codes=tuple(
            str(ordered_zip_codes[facility_idx])
            for facility_idx in selected_indices_tuple
        ),
        covered_zip_codes=tuple(
            str(ordered_zip_codes[demand_idx]) for demand_idx in covered_indices
        ),
        selected_facility_indices=selected_indices_tuple,
        covered_zone_indices=covered_indices,
        solver_status=solver_status,
        optimal=bool(optimal),
        mip_gap=mip_gap,
        solver_name=solver_name,
        optimization_solver=normalize_optimization_solver(optimization_solver),
        solver_runtime_seconds=solver_runtime_seconds,
        risk_objective_value=risk_objective_value,
    )


def _build_solver_params(config: EmsExactShapConfig) -> dict[str, float | int | str]:
    return {
        "Threads": config.gurobi_threads,
        "Seed": config.solver_seed,
        "MIPGap": config.mip_gap,
        "MIPGapAbs": config.mip_gap_abs,
        "FeasibilityTol": 1e-9,
    }


def _build_deterministic_coverage_tie_break_expression(
    *,
    x: pyo.Var,
    y: pyo.Var,
    demand_count: int,
    facility_count: int,
) -> pyo.Expression:
    """Stable secondary objective for the paper's deterministic tie-break assumption."""

    coverage_scale = 1.0 / (max(1, demand_count) + 1.0)
    return -(
        sum(
            (facility_idx + 1) * x[facility_idx]
            for facility_idx in range(facility_count)
        )
        + coverage_scale
        * sum(
            (demand_idx + 1) * y[demand_idx]
            for demand_idx in range(demand_count)
        )
    )


def _realized_coverage_value(
    covered_zone_indices: Sequence[int],
    true_demand: Sequence[float] | np.ndarray,
) -> float:
    return _coverage_fraction(
        _covered_demand_value(covered_zone_indices, true_demand),
        _total_demand(true_demand),
    )


def _covered_demand_value(
    covered_zone_indices: Sequence[int],
    demand: Sequence[float] | np.ndarray,
) -> float:
    demand_array = np.maximum(np.asarray(demand, dtype=float), 0.0)
    return float(sum(demand_array[int(zone_idx)] for zone_idx in covered_zone_indices))


def _total_demand(demand: Sequence[float] | np.ndarray) -> float:
    return float(np.maximum(np.asarray(demand, dtype=float), 0.0).sum())


def _coverage_fraction(covered_demand: float, total_demand: float) -> float:
    if total_demand <= 0.0:
        return 0.0
    return float(covered_demand / total_demand)


def _coverage_weight(demand: float, total_demand: float) -> float:
    if total_demand <= 0.0:
        return 0.0
    return float(max(demand, 0.0) / total_demand)


def _build_coverage_solution_row(
    timestamp_hour: str,
    solution_type: str,
    solution: MaximumCoverageResult,
    true_demand: Sequence[float] | np.ndarray,
    config: EmsExactShapConfig,
) -> dict[str, Any]:
    realized_covered_demand = _covered_demand_value(
        solution.covered_zone_indices,
        true_demand,
    )
    actual_total_demand = _total_demand(true_demand)
    return {
        "timestamp_hour": timestamp_hour,
        "solution_type": solution_type,
        "objective_value": float(solution.objective_value),
        "predicted_covered_demand": float(solution.covered_demand),
        "predicted_total_demand": float(solution.total_demand),
        "realized_ems_value": _coverage_fraction(
            realized_covered_demand,
            actual_total_demand,
        ),
        "realized_covered_demand": realized_covered_demand,
        "actual_total_demand": actual_total_demand,
        "selected_facility_zip_codes": json.dumps(
            list(solution.selected_facility_zip_codes)
        ),
        "covered_zip_codes": json.dumps(list(solution.covered_zip_codes)),
        "selected_facility_count": len(solution.selected_facility_zip_codes),
        "covered_zone_count": len(solution.covered_zip_codes),
        "coverage_radius_km": float(config.coverage_radius_km),
        "facility_budget": int(config.facility_budget),
        "coverage_solver": solution.solver_name,
        "optimization_solver": solution.optimization_solver,
        "solver_status": solution.solver_status,
        "optimal": bool(solution.optimal),
        "mip_gap": solution.mip_gap,
        "solver_runtime_seconds": solution.solver_runtime_seconds,
        "risk_objective_value": solution.risk_objective_value,
    }


def _sum_solver_runtime_seconds(*solve_results: Any) -> float | None:
    runtimes = [
        result.solver_runtime_seconds
        for result in solve_results
        if result.solver_runtime_seconds is not None
    ]
    if not runtimes:
        return None
    return float(sum(runtimes))


def _maximum_coverage_decision_changed(
    left_solution: MaximumCoverageResult,
    right_solution: MaximumCoverageResult,
) -> bool:
    return set(left_solution.selected_facility_indices) != set(
        right_solution.selected_facility_indices
    )


def _build_coalition_rows(
    timestamp_hour: str,
    predictive_values: np.ndarray,
    decision_values: np.ndarray,
    decision_solutions: Sequence[MaximumCoverageResult],
    player_names: Sequence[str],
    ante_decision_values: np.ndarray | None = None,
    cvar_decision_values: np.ndarray | None = None,
    cvar_decision_solutions: Sequence[MaximumCoverageResult] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(decision_solutions) != len(predictive_values):
        raise ValueError("decision_solutions must align with coalition values.")
    if cvar_decision_solutions is not None and len(cvar_decision_solutions) != len(
        predictive_values
    ):
        raise ValueError("cvar_decision_solutions must align with coalition values.")
    for coalition_mask in range(len(predictive_values)):
        included_players = [
            player_name
            for player_idx, player_name in enumerate(player_names)
            if coalition_mask & (1 << player_idx)
        ]
        rows.append(
            {
                "timestamp_hour": timestamp_hour,
                "coalition_mask": coalition_mask,
                "subset_size": int(coalition_mask.bit_count()),
                "included_players": json.dumps(included_players),
                "predictive_value": float(predictive_values[coalition_mask]),
                "decision_value": float(decision_values[coalition_mask]),
                "decision_characteristic_value": float(
                    decision_values[coalition_mask] - decision_values[0]
                ),
                "decision_selected_facility_zip_codes": json.dumps(
                    list(decision_solutions[coalition_mask].selected_facility_zip_codes)
                ),
                "decision_selected_facility_indices": json.dumps(
                    list(decision_solutions[coalition_mask].selected_facility_indices)
                ),
            }
        )
        if ante_decision_values is not None:
            rows[-1]["ante_decision_value"] = float(
                ante_decision_values[coalition_mask]
            )
            rows[-1]["ante_decision_characteristic_value"] = float(
                ante_decision_values[coalition_mask] - ante_decision_values[0]
            )
        if cvar_decision_values is not None:
            rows[-1]["cvar_decision_value"] = float(
                cvar_decision_values[coalition_mask]
            )
            rows[-1]["cvar_decision_characteristic_value"] = float(
                cvar_decision_values[coalition_mask] - cvar_decision_values[0]
            )
        if cvar_decision_solutions is not None:
            rows[-1]["cvar_decision_selected_facility_zip_codes"] = json.dumps(
                list(
                    cvar_decision_solutions[
                        coalition_mask
                    ].selected_facility_zip_codes
                )
            )
            rows[-1]["cvar_decision_selected_facility_indices"] = json.dumps(
                list(cvar_decision_solutions[coalition_mask].selected_facility_indices)
            )
    return rows


def _build_summary_shap_frame(
    hourly_shap: pd.DataFrame,
    player_names: Sequence[str],
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    predictive_abs_means: dict[str, float] = {}
    decision_abs_means: dict[str, float] = {}
    ante_available = all(
        f"ante_decision_shap_{player_name}" in hourly_shap.columns
        for player_name in player_names
    )
    ante_abs_means: dict[str, float] = {}
    for player_name in player_names:
        predictive_column = hourly_shap[f"predictive_shap_{player_name}"]
        decision_column = hourly_shap[f"decision_shap_{player_name}"]
        predictive_abs_means[player_name] = float(predictive_column.abs().mean())
        decision_abs_means[player_name] = float(decision_column.abs().mean())
        if ante_available:
            ante_column = hourly_shap[f"ante_decision_shap_{player_name}"]
            ante_abs_means[player_name] = float(ante_column.abs().mean())

    predictive_ranks = _descending_rank_map(predictive_abs_means)
    decision_ranks = _descending_rank_map(decision_abs_means)
    ante_ranks = _descending_rank_map(ante_abs_means) if ante_available else {}
    for player_name in player_names:
        predictive_column = hourly_shap[f"predictive_shap_{player_name}"]
        decision_column = hourly_shap[f"decision_shap_{player_name}"]
        activation_column = hourly_shap[f"decision_activation_rate_{player_name}"]
        activated_value_sum_column = hourly_shap[
            f"decision_activated_value_sum_{player_name}"
        ]
        activation_rate_sum = float(activation_column.sum())
        decision_activated_value = (
            0.0
            if activation_rate_sum <= 0.0
            else float(activated_value_sum_column.sum() / activation_rate_sum)
        )
        row = {
            "feature": player_name,
            "predictive_mean_signed_shap": float(predictive_column.mean()),
            "predictive_mean_abs_shap": predictive_abs_means[player_name],
            "predictive_rank": predictive_ranks[player_name],
            "decision_mean_signed_shap": float(decision_column.mean()),
            "decision_mean_abs_shap": decision_abs_means[player_name],
            "decision_rank": decision_ranks[player_name],
            "decision_activation_rate": float(activation_column.mean()),
            "decision_activated_value": decision_activated_value,
        }
        if ante_available:
            ante_column = hourly_shap[f"ante_decision_shap_{player_name}"]
            row.update(
                {
                    "ante_decision_mean_signed_shap": float(ante_column.mean()),
                    "ante_decision_mean_abs_shap": ante_abs_means[player_name],
                    "ante_decision_rank": ante_ranks[player_name],
                }
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values("predictive_rank").reset_index(drop=True)


def _build_cvar_summary_shap_frame(
    hourly_shap: pd.DataFrame,
    player_names: Sequence[str],
) -> pd.DataFrame:
    required_prefix = "cvar_decision_shap_"
    if not all(
        f"{required_prefix}{player_name}" in hourly_shap.columns
        for player_name in player_names
    ):
        return pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    deterministic_abs_means: dict[str, float] = {}
    cvar_abs_means: dict[str, float] = {}
    for player_name in player_names:
        deterministic_column = hourly_shap[f"decision_shap_{player_name}"]
        cvar_column = hourly_shap[f"cvar_decision_shap_{player_name}"]
        deterministic_abs_means[player_name] = float(deterministic_column.abs().mean())
        cvar_abs_means[player_name] = float(cvar_column.abs().mean())

    deterministic_ranks = _descending_rank_map(deterministic_abs_means)
    cvar_ranks = _descending_rank_map(cvar_abs_means)
    for player_name in player_names:
        deterministic_column = hourly_shap[f"decision_shap_{player_name}"]
        cvar_column = hourly_shap[f"cvar_decision_shap_{player_name}"]
        cvar_activation_column = hourly_shap[
            f"cvar_decision_activation_rate_{player_name}"
        ]
        cvar_activated_value_sum_column = hourly_shap[
            f"cvar_decision_activated_value_sum_{player_name}"
        ]
        cvar_activation_rate_sum = float(cvar_activation_column.sum())
        cvar_decision_activated_value = (
            0.0
            if cvar_activation_rate_sum <= 0.0
            else float(cvar_activated_value_sum_column.sum() / cvar_activation_rate_sum)
        )
        summary_rows.append(
            {
                "feature": player_name,
                "decision_mean_signed_shap": float(deterministic_column.mean()),
                "decision_mean_abs_shap": deterministic_abs_means[player_name],
                "decision_rank": deterministic_ranks[player_name],
                "cvar_decision_mean_signed_shap": float(cvar_column.mean()),
                "cvar_decision_mean_abs_shap": cvar_abs_means[player_name],
                "cvar_decision_rank": cvar_ranks[player_name],
                "cvar_decision_activation_rate": float(
                    cvar_activation_column.mean()
                ),
                "cvar_decision_activated_value": cvar_decision_activated_value,
                "cvar_minus_decision_mean_signed_shap": float(
                    cvar_column.mean() - deterministic_column.mean()
                ),
                "cvar_minus_decision_mean_abs_shap": float(
                    cvar_column.abs().mean() - deterministic_column.abs().mean()
                ),
            }
        )
    return (
        pd.DataFrame(summary_rows)
        .sort_values("cvar_decision_rank")
        .reset_index(drop=True)
    )


def _descending_rank_map(values_by_feature: dict[str, float]) -> dict[str, int]:
    ranks = (
        pd.Series(values_by_feature, dtype=float)
        .rank(method="dense", ascending=False)
        .astype(int)
        .to_dict()
    )
    return {feature_name: int(rank) for feature_name, rank in ranks.items()}


def _normalize_decision_shap_sample_counts(
    sample_counts: Sequence[int],
    config_name: str,
) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for sample_count in sample_counts:
        count = _validate_positive_sample_count(sample_count, config_name)
        if count in seen:
            continue
        seen.add(count)
        normalized.append(count)
    return tuple(normalized)


def _validate_positive_sample_count(sample_count: int, name: str) -> int:
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise ValueError(f"{name} must contain positive integers.")
    count = int(sample_count)
    if count <= 0:
        raise ValueError(f"{name} must contain positive integers.")
    return count


def _resolve_decision_shap_seed(
    seed: int | None,
    fallback_seed: int,
    config_name: str,
) -> int:
    resolved_seed = fallback_seed if seed is None else seed
    if isinstance(resolved_seed, bool) or not isinstance(
        resolved_seed,
        (int, np.integer),
    ):
        raise ValueError(f"{config_name} must be an integer when provided.")
    return int(resolved_seed)


def _validate_scalar_approximation_game(
    coalition_values: Sequence[float] | np.ndarray,
    feature_count: int,
    method_name: str,
) -> np.ndarray:
    if feature_count < 0:
        raise ValueError("feature_count must be non-negative.")
    expected_coalition_count = 1 << feature_count
    coalition_array = np.asarray(coalition_values, dtype=float)
    if coalition_array.shape != (expected_coalition_count,):
        raise ValueError(
            f"{method_name} requires one scalar value per coalition. "
            f"Expected shape {(expected_coalition_count,)}, got "
            f"{coalition_array.shape}."
        )
    return coalition_array


def _shapley_kernel_weight(feature_count: int, subset_size: int) -> float:
    if subset_size <= 0 or subset_size >= feature_count:
        raise ValueError("Shapley kernel is defined only for interior coalitions.")
    return float(
        (feature_count - 1)
        / (
            math.comb(feature_count, subset_size)
            * subset_size
            * (feature_count - subset_size)
        )
    )


def _decision_shap_seed_sequence(
    random_state: int,
    hour_idx: int,
    method_id: int,
    sample_count: int,
) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [
            int(random_state),
            int(hour_idx),
            int(method_id),
            int(sample_count),
        ]
    )


def _subset_weights(feature_count: int) -> tuple[float, ...]:
    denominator = math.factorial(feature_count)
    return tuple(
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / denominator
        for subset_size in range(feature_count)
    )


def _resolve_observation_array(
    observation: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    if isinstance(observation, pd.Series):
        return observation.loc[list(feature_names)].to_numpy(dtype=float, copy=True)
    if isinstance(observation, pd.DataFrame):
        if len(observation) != 1:
            raise ValueError("observation DataFrame must contain exactly one row.")
        return observation.loc[:, list(feature_names)].iloc[0].to_numpy(
            dtype=float,
            copy=True,
        )
    array = np.asarray(observation, dtype=float)
    if array.shape != (len(feature_names),):
        raise ValueError(
            "observation must have one value per feature. "
            f"Expected shape {(len(feature_names),)}, got {array.shape}."
        )
    return array
