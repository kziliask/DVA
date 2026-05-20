from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack
from xgboost import XGBRegressor

from dva.analysis.ems_exact_shap import (
    DEFAULT_DISTANCE_MATRIX_PATH,
    DEFAULT_EXCLUDED_ZIP_CODES,
    DEFAULT_METADATA_PATH,
    DEFAULT_OBJECTIVE_TOLERANCE,
    DEFAULT_TEST_MONTHS,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_ZONE_ORDER_PATH,
    EMS_TIMESTAMP_COLUMN,
    EmsExactShapConfig,
    _as_float32_matrix,
    _build_time_split,
    _build_xgb_params,
    _load_distance_matrix,
    _load_ems_frames,
    _load_zone_order,
    _maybe_sample_training_rows,
    _resolve_feature_columns,
    _sample_background_frame,
    _sample_explanation_hours,
    _target_columns,
    _total_demand,
    build_coverage_matrix,
)
from dva.case_studies.ems.models import (
    EMS_XGB_MODEL_IDS,
    ems_xgb_config_kwargs,
    make_ems_xgb_model_manifest,
    resolve_ems_xgb_model_record,
)


DEFAULT_OUTDIR = Path("results/ems/baseline_experiment")
DEFAULT_REGIMES = ((1.0, 3), (1.0, 5), (1.0, 8), (2.0, 5))
DEFAULT_RANDOM_STATE = 0
DEFAULT_TIME_LIMIT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class Regime:
    tau: float
    p: int

    @property
    def label(self) -> str:
        return f"tau{self.tau:g}_p{self.p}"


@dataclass(frozen=True, slots=True)
class CoverageSolution:
    objective_value: float
    covered_demand: float
    total_demand: float
    selected_indices: tuple[int, ...]
    covered_indices: tuple[int, ...]
    selected_zip_codes: tuple[str, ...]
    covered_zip_codes: tuple[str, ...]
    primary_objective_value: float
    solver_success: bool
    solver_message: str


class ScipyMaximumCoverageSolver:
    def __init__(
        self,
        *,
        coverage_matrix: np.ndarray,
        zip_codes: Sequence[str],
        facility_budget: int,
        objective_tolerance: float,
        time_limit_seconds: float,
    ) -> None:
        self.coverage_matrix = np.asarray(coverage_matrix, dtype=bool)
        self.zip_codes = tuple(str(zip_code) for zip_code in zip_codes)
        self.facility_budget = int(facility_budget)
        self.objective_tolerance = float(objective_tolerance)
        self.time_limit_seconds = float(time_limit_seconds)
        self.demand_count, self.facility_count = self.coverage_matrix.shape
        if self.demand_count != len(self.zip_codes):
            raise ValueError("coverage_matrix demand rows must align with zip_codes.")

        self.variable_count = self.facility_count + self.demand_count
        self.integrality = np.ones(self.variable_count, dtype=int)
        self.bounds = Bounds(
            np.zeros(self.variable_count, dtype=float),
            np.ones(self.variable_count, dtype=float),
        )
        self.base_matrix, self.base_lb, self.base_ub = self._build_base_constraints()
        self.tie_break_objective = self._build_tie_break_objective()

    def solve(self, demand: Sequence[float] | np.ndarray) -> CoverageSolution:
        demand_array = np.maximum(np.asarray(demand, dtype=float), 0.0)
        if demand_array.shape != (self.demand_count,):
            raise ValueError(
                "demand must be one-dimensional and aligned with coverage_matrix."
            )

        total_demand = _total_demand(demand_array)
        primary_weights = (
            demand_array / total_demand
            if total_demand > 0.0
            else np.zeros_like(demand_array, dtype=float)
        )
        primary_objective = np.r_[
            np.zeros(self.facility_count, dtype=float),
            primary_weights,
        ]
        primary_result = milp(
            c=-primary_objective,
            integrality=self.integrality,
            bounds=self.bounds,
            constraints=LinearConstraint(
                self.base_matrix,
                self.base_lb,
                self.base_ub,
            ),
            options=self._solver_options(),
        )
        if not primary_result.success:
            return self._failed_solution(demand_array, total_demand, primary_result)

        primary_value = float(-primary_result.fun)
        tie_matrix = vstack(
            [
                self.base_matrix,
                -primary_objective.reshape(1, -1),
            ],
            format="csr",
        )
        tie_lb = np.r_[self.base_lb, -np.inf]
        tie_ub = np.r_[self.base_ub, -(primary_value - self.objective_tolerance)]
        tie_result = milp(
            c=self.tie_break_objective,
            integrality=self.integrality,
            bounds=self.bounds,
            constraints=LinearConstraint(tie_matrix, tie_lb, tie_ub),
            options=self._solver_options(),
        )
        if not tie_result.success:
            return self._failed_solution(demand_array, total_demand, tie_result)

        selected_indices = tuple(
            int(idx)
            for idx in np.flatnonzero(tie_result.x[: self.facility_count] > 0.5)
        )
        return self._solution_from_selected_indices(
            demand_array,
            total_demand,
            selected_indices,
            primary_value,
            True,
            str(tie_result.message),
        )

    def evaluate_selected_indices(
        self,
        demand: Sequence[float] | np.ndarray,
        selected_indices: Sequence[int],
        *,
        primary_objective_value: float | None = None,
    ) -> CoverageSolution:
        demand_array = np.maximum(np.asarray(demand, dtype=float), 0.0)
        total_demand = _total_demand(demand_array)
        return self._solution_from_selected_indices(
            demand_array,
            total_demand,
            tuple(int(index) for index in selected_indices),
            (
                float(primary_objective_value)
                if primary_objective_value is not None
                else float("nan")
            ),
            True,
            "evaluated fixed selected_indices",
        )

    def _build_base_constraints(self) -> tuple[Any, np.ndarray, np.ndarray]:
        matrix = lil_matrix((self.demand_count + 1, self.variable_count), dtype=float)
        for demand_idx in range(self.demand_count):
            matrix[demand_idx, : self.facility_count] = -self.coverage_matrix[
                demand_idx
            ].astype(float)
            matrix[demand_idx, self.facility_count + demand_idx] = 1.0
        matrix[self.demand_count, : self.facility_count] = 1.0
        lower = np.full(self.demand_count + 1, -np.inf, dtype=float)
        upper = np.r_[np.zeros(self.demand_count, dtype=float), self.facility_budget]
        return matrix.tocsr(), lower, upper

    def _build_tie_break_objective(self) -> np.ndarray:
        facility_weights = np.arange(1, self.facility_count + 1, dtype=float)
        covered_weights = (
            np.arange(1, self.demand_count + 1, dtype=float)
            / (max(1, self.demand_count) + 1.0)
        )
        return np.r_[facility_weights, covered_weights]

    def _solver_options(self) -> dict[str, float | bool]:
        return {
            "mip_rel_gap": 0.0,
            "time_limit": self.time_limit_seconds,
            "disp": False,
        }

    def _solution_from_selected_indices(
        self,
        demand_array: np.ndarray,
        total_demand: float,
        selected_indices: Sequence[int],
        primary_objective_value: float,
        solver_success: bool,
        solver_message: str,
    ) -> CoverageSolution:
        selected = tuple(sorted({int(index) for index in selected_indices}))
        if selected:
            covered_mask = np.asarray(
                self.coverage_matrix[:, list(selected)].any(axis=1),
                dtype=bool,
            )
        else:
            covered_mask = np.zeros(self.demand_count, dtype=bool)
        covered_indices = tuple(int(idx) for idx in np.flatnonzero(covered_mask))
        covered_demand = float(demand_array[list(covered_indices)].sum())
        objective_value = covered_demand / total_demand if total_demand > 0.0 else 0.0
        return CoverageSolution(
            objective_value=float(objective_value),
            covered_demand=covered_demand,
            total_demand=float(total_demand),
            selected_indices=selected,
            covered_indices=covered_indices,
            selected_zip_codes=tuple(self.zip_codes[idx] for idx in selected),
            covered_zip_codes=tuple(self.zip_codes[idx] for idx in covered_indices),
            primary_objective_value=float(primary_objective_value),
            solver_success=bool(solver_success),
            solver_message=str(solver_message),
        )

    def _failed_solution(
        self,
        demand_array: np.ndarray,
        total_demand: float,
        result: Any,
    ) -> CoverageSolution:
        return self._solution_from_selected_indices(
            demand_array,
            total_demand,
            (),
            float("nan"),
            False,
            str(result.message),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EMS oracle, y_null baseline, and full-model decisions on "
            "the holdout hours not used by the 100-hour SHAP eval set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--zone-order-path", type=Path, default=DEFAULT_ZONE_ORDER_PATH)
    parser.add_argument(
        "--distance-matrix-path",
        type=Path,
        default=DEFAULT_DISTANCE_MATRIX_PATH,
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        choices=EMS_XGB_MODEL_IDS,
        help="EMS XGB model id to evaluate. Defaults to all 25 EMS XGB models.",
    )
    parser.add_argument(
        "--regime",
        action="append",
        default=None,
        metavar="TAU:P",
        help="Regime to evaluate, e.g. 1:3. Defaults to the four paper regimes.",
    )
    parser.add_argument("--holdout-hours", type=int, default=100)
    parser.add_argument("--test-months", type=int, default=DEFAULT_TEST_MONTHS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--background-rows", type=int, default=100)
    parser.add_argument("--train-sample-rows", type=int, default=None)
    parser.add_argument(
        "--objective-tolerance",
        type=float,
        default=DEFAULT_OBJECTIVE_TOLERANCE,
    )
    parser.add_argument(
        "--solver-time-limit-seconds",
        type=float,
        default=DEFAULT_TIME_LIMIT_SECONDS,
    )
    parser.add_argument(
        "--max-rest-hours",
        type=int,
        default=None,
        help="Optional cap for smoke tests; evaluates the first N unsampled holdout hours.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing CSV/JSON outputs in the output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started_at = time.perf_counter()
    regimes = _parse_regimes(args.regime)
    model_ids = tuple(args.model_id or EMS_XGB_MODEL_IDS)
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.outdir} already contains outputs; pass --overwrite to replace them."
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    base_config = EmsExactShapConfig(
        x_path=args.x_path,
        y_path=args.y_path,
        metadata_path=args.metadata_path,
        zone_order_path=args.zone_order_path,
        distance_matrix_path=args.distance_matrix_path,
        outdir=args.outdir,
        holdout_hours=args.holdout_hours,
        test_months=args.test_months,
        background_rows=args.background_rows,
        random_state=args.random_state,
        train_sample_rows=args.train_sample_rows,
        excluded_zip_codes=DEFAULT_EXCLUDED_ZIP_CODES,
        objective_tolerance=args.objective_tolerance,
        compute_cvar_decision_shap=False,
    )
    prepared = _prepare_data(base_config, args.max_rest_hours)
    solvers = _build_solvers(
        base_config,
        regimes,
        prepared["zip_codes"],
        time_limit_seconds=args.solver_time_limit_seconds,
    )

    oracle = _compute_oracle_rows(
        solvers=solvers,
        regimes=regimes,
        rest_y=prepared["rest_y"],
        target_columns=prepared["target_columns"],
        rest_index=prepared["rest_index"],
    )
    oracle_by_regime_and_position = {
        (row["regime"], int(row["holdout_position"])): row for row in oracle
    }
    pd.DataFrame(oracle).to_csv(args.outdir / "oracle_objectives.csv", index=False)

    per_hour_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for model_idx, model_id in enumerate(model_ids, start=1):
        model_started_at = time.perf_counter()
        print(
            f"[{model_idx}/{len(model_ids)}] training/evaluating {model_id}",
            flush=True,
        )
        model_config, model_record = _model_config(base_config, model_id)
        model = _fit_model(
            train_x=prepared["train_x"],
            train_y=prepared["train_y"],
            target_columns=prepared["target_columns"],
            feature_columns=prepared["feature_columns"],
            config=model_config,
        )
        background_predictions = _predict_background_demand(
            model=model,
            train_x=prepared["train_x"],
            feature_columns=prepared["feature_columns"],
            config=model_config,
        )
        full_predictions = _predict_full_demand(
            model,
            prepared["rest_x"],
            prepared["feature_columns"],
        )
        for regime in regimes:
            solver = solvers[regime.label]
            baseline_solution = solver.solve(background_predictions)
            baseline_rows.append(
                _build_baseline_decision_row(
                    model_id=model_id,
                    model_record=model_record,
                    regime=regime,
                    solution=baseline_solution,
                )
            )
            for row_idx, rest_row in prepared["rest_index"].iterrows():
                true_demand = prepared["rest_y"].iloc[row_idx][
                    list(prepared["target_columns"])
                ].to_numpy(dtype=float, copy=True)
                baseline_realized = solver.evaluate_selected_indices(
                    true_demand,
                    baseline_solution.selected_indices,
                    primary_objective_value=baseline_solution.primary_objective_value,
                )
                full_solution = solver.solve(full_predictions[row_idx])
                full_realized = solver.evaluate_selected_indices(
                    true_demand,
                    full_solution.selected_indices,
                    primary_objective_value=full_solution.primary_objective_value,
                )
                oracle_row = oracle_by_regime_and_position[
                    (regime.label, int(rest_row["holdout_position"]))
                ]
                per_hour_rows.append(
                    _build_per_hour_row(
                        model_id=model_id,
                        model_record=model_record,
                        regime=regime,
                        rest_row=rest_row,
                        oracle_row=oracle_row,
                        baseline_predicted=baseline_solution,
                        baseline_realized=baseline_realized,
                        full_predicted=full_solution,
                        full_realized=full_realized,
                    )
                )
        print(
            f"[{model_idx}/{len(model_ids)}] finished {model_id} in "
            f"{time.perf_counter() - model_started_at:.2f}s",
            flush=True,
        )

    per_hour = pd.DataFrame(per_hour_rows)
    baseline_decisions = pd.DataFrame(baseline_rows)
    summary = _build_summary(per_hour)
    per_hour.to_csv(args.outdir / "per_hour_objectives.csv", index=False)
    baseline_decisions.to_csv(args.outdir / "baseline_decisions.csv", index=False)
    summary.to_csv(args.outdir / "summary_by_model_regime.csv", index=False)
    make_ems_xgb_model_manifest().loc[
        lambda frame: frame["model_id"].isin(model_ids)
    ].to_csv(args.outdir / "model_manifest.csv", index=False)
    _write_metadata(
        args.outdir,
        args=args,
        regimes=regimes,
        model_ids=model_ids,
        prepared=prepared,
        runtime_seconds=time.perf_counter() - started_at,
    )
    print(f"Wrote EMS baseline experiment outputs to {args.outdir}", flush=True)


def _parse_regimes(raw_regimes: Sequence[str] | None) -> tuple[Regime, ...]:
    if not raw_regimes:
        return tuple(Regime(tau=tau, p=p) for tau, p in DEFAULT_REGIMES)
    regimes = []
    for raw in raw_regimes:
        if ":" not in raw:
            raise ValueError(f"Regime must use TAU:P format, got {raw!r}.")
        tau_text, p_text = raw.split(":", maxsplit=1)
        regimes.append(Regime(tau=float(tau_text), p=int(p_text)))
    return tuple(regimes)


def _prepare_data(
    config: EmsExactShapConfig,
    max_rest_hours: int | None,
) -> dict[str, Any]:
    x_frame, y_frame, metadata = _load_ems_frames(config)
    zone_order = _load_zone_order(config.zone_order_path, tuple(_target_columns(y_frame)))
    target_columns = tuple(zone_order["target_column"].astype(str))
    zip_codes = tuple(zone_order["zip_code"].astype(str))
    y_frame = y_frame.loc[:, [EMS_TIMESTAMP_COLUMN, *target_columns]].copy()
    feature_columns = _resolve_feature_columns(x_frame, metadata)
    time_split = _build_time_split(
        x_frame.loc[:, [EMS_TIMESTAMP_COLUMN, *feature_columns]],
        y_frame,
        test_months=config.test_months,
    )
    train_x, train_y = _maybe_sample_training_rows(
        time_split.train_x,
        time_split.train_y,
        train_sample_rows=config.train_sample_rows,
        random_state=config.random_state,
    )
    _, _, explained_rows = _sample_explanation_hours(
        time_split.holdout_x,
        time_split.holdout_y,
        time_split.holdout_source_rows,
        holdout_hours=config.holdout_hours,
        max_hours=None,
        random_state=config.random_state,
    )
    sampled_positions = {int(row["holdout_position"]) for row in explained_rows}
    rest_positions = [
        position
        for position in range(len(time_split.holdout_x))
        if position not in sampled_positions
    ]
    if max_rest_hours is not None:
        rest_positions = rest_positions[: int(max_rest_hours)]
    rest_x = time_split.holdout_x.iloc[rest_positions].reset_index(drop=True)
    rest_y = time_split.holdout_y.iloc[rest_positions].reset_index(drop=True)
    rest_index = pd.DataFrame(
        {
            "row_idx": np.arange(len(rest_positions), dtype=int),
            "holdout_position": rest_positions,
            "source_row_position": [
                int(time_split.holdout_source_rows[position])
                for position in rest_positions
            ],
            "timestamp_hour": rest_x[EMS_TIMESTAMP_COLUMN].astype(str).tolist(),
        }
    )
    print(
        f"Prepared EMS holdout rest set: {len(rest_positions):,} hours "
        f"outside the {len(sampled_positions):,} sampled SHAP hours.",
        flush=True,
    )
    return {
        "x_frame": x_frame,
        "y_frame": y_frame,
        "metadata": metadata,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "zip_codes": zip_codes,
        "time_split": time_split,
        "train_x": train_x,
        "train_y": train_y,
        "rest_x": rest_x,
        "rest_y": rest_y,
        "rest_index": rest_index,
        "explained_rows": explained_rows,
    }


def _build_solvers(
    config: EmsExactShapConfig,
    regimes: Sequence[Regime],
    zip_codes: Sequence[str],
    *,
    time_limit_seconds: float,
) -> dict[str, ScipyMaximumCoverageSolver]:
    distance_matrix = _load_distance_matrix(config.distance_matrix_path)
    solvers = {}
    for regime in regimes:
        coverage_matrix = build_coverage_matrix(
            distance_matrix,
            zip_codes,
            coverage_radius_km=regime.tau,
        )
        solvers[regime.label] = ScipyMaximumCoverageSolver(
            coverage_matrix=coverage_matrix,
            zip_codes=zip_codes,
            facility_budget=regime.p,
            objective_tolerance=config.objective_tolerance,
            time_limit_seconds=time_limit_seconds,
        )
    return solvers


def _model_config(
    base_config: EmsExactShapConfig,
    model_id: str,
) -> tuple[EmsExactShapConfig, dict[str, Any]]:
    record = resolve_ems_xgb_model_record(model_id)
    config = EmsExactShapConfig(
        x_path=base_config.x_path,
        y_path=base_config.y_path,
        metadata_path=base_config.metadata_path,
        zone_order_path=base_config.zone_order_path,
        distance_matrix_path=base_config.distance_matrix_path,
        outdir=base_config.outdir,
        holdout_hours=base_config.holdout_hours,
        test_months=base_config.test_months,
        background_rows=base_config.background_rows,
        random_state=base_config.random_state,
        train_sample_rows=base_config.train_sample_rows,
        objective_tolerance=base_config.objective_tolerance,
        model_id=model_id,
        **ems_xgb_config_kwargs(record),
        compute_cvar_decision_shap=False,
    )
    return config, record


def _fit_model(
    *,
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    target_columns: Sequence[str],
    feature_columns: Sequence[str],
    config: EmsExactShapConfig,
) -> XGBRegressor:
    model = XGBRegressor(**_build_xgb_params(config))
    model.fit(
        _as_float32_matrix(train_x.loc[:, list(feature_columns)].to_numpy(dtype=np.float32)),
        _as_float32_matrix(
            train_y.loc[:, list(target_columns)].to_numpy(dtype=np.float32, copy=True)
        ),
    )
    return model


def _predict_background_demand(
    *,
    model: XGBRegressor,
    train_x: pd.DataFrame,
    feature_columns: Sequence[str],
    config: EmsExactShapConfig,
) -> np.ndarray:
    background = _sample_background_frame(
        train_frame=train_x,
        feature_columns=feature_columns,
        rows=config.background_rows,
        random_state=config.random_state,
    )
    predictions = np.asarray(
        model.predict(
            _as_float32_matrix(background.to_numpy(dtype=np.float32, copy=True))
        ),
        dtype=float,
    )
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
    return np.maximum(predictions.mean(axis=0), 0.0)


def _predict_full_demand(
    model: XGBRegressor,
    rest_x: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    predictions = np.asarray(
        model.predict(
            _as_float32_matrix(
                rest_x.loc[:, list(feature_columns)].to_numpy(dtype=np.float32, copy=True)
            )
        ),
        dtype=float,
    )
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
    return np.maximum(predictions, 0.0)


def _compute_oracle_rows(
    *,
    solvers: dict[str, ScipyMaximumCoverageSolver],
    regimes: Sequence[Regime],
    rest_y: pd.DataFrame,
    target_columns: Sequence[str],
    rest_index: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime in regimes:
        solver = solvers[regime.label]
        started_at = time.perf_counter()
        for row_idx, rest_row in rest_index.iterrows():
            true_demand = rest_y.iloc[row_idx][list(target_columns)].to_numpy(
                dtype=float,
                copy=True,
            )
            solution = solver.solve(true_demand)
            rows.append(
                {
                    "regime": regime.label,
                    "tau": regime.tau,
                    "p": regime.p,
                    "timestamp_hour": rest_row["timestamp_hour"],
                    "holdout_position": int(rest_row["holdout_position"]),
                    "source_row_position": int(rest_row["source_row_position"]),
                    "oracle_objective_value": solution.objective_value,
                    "oracle_covered_demand": solution.covered_demand,
                    "actual_total_demand": solution.total_demand,
                    "oracle_selected_zip_codes": json.dumps(
                        list(solution.selected_zip_codes)
                    ),
                    "oracle_covered_zip_codes": json.dumps(
                        list(solution.covered_zip_codes)
                    ),
                    "oracle_solver_success": solution.solver_success,
                    "oracle_solver_message": solution.solver_message,
                }
            )
        print(
            f"Computed oracle objectives for {regime.label} in "
            f"{time.perf_counter() - started_at:.2f}s",
            flush=True,
        )
    return rows


def _build_baseline_decision_row(
    *,
    model_id: str,
    model_record: dict[str, Any],
    regime: Regime,
    solution: CoverageSolution,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "regime": regime.label,
        "tau": regime.tau,
        "p": regime.p,
        **_model_record_columns(model_record),
        "baseline_predicted_objective_value": solution.objective_value,
        "baseline_predicted_covered_demand": solution.covered_demand,
        "baseline_predicted_total_demand": solution.total_demand,
        "baseline_selected_zip_codes": json.dumps(list(solution.selected_zip_codes)),
        "baseline_covered_zip_codes": json.dumps(list(solution.covered_zip_codes)),
        "baseline_solver_success": solution.solver_success,
        "baseline_solver_message": solution.solver_message,
    }


def _build_per_hour_row(
    *,
    model_id: str,
    model_record: dict[str, Any],
    regime: Regime,
    rest_row: pd.Series,
    oracle_row: dict[str, Any],
    baseline_predicted: CoverageSolution,
    baseline_realized: CoverageSolution,
    full_predicted: CoverageSolution,
    full_realized: CoverageSolution,
) -> dict[str, Any]:
    oracle_objective = float(oracle_row["oracle_objective_value"])
    return {
        "model_id": model_id,
        "regime": regime.label,
        "tau": regime.tau,
        "p": regime.p,
        **_model_record_columns(model_record),
        "timestamp_hour": rest_row["timestamp_hour"],
        "holdout_position": int(rest_row["holdout_position"]),
        "source_row_position": int(rest_row["source_row_position"]),
        "actual_total_demand": float(oracle_row["actual_total_demand"]),
        "oracle_objective_value": oracle_objective,
        "oracle_covered_demand": float(oracle_row["oracle_covered_demand"]),
        "baseline_predicted_objective_value": baseline_predicted.objective_value,
        "baseline_predicted_covered_demand": baseline_predicted.covered_demand,
        "baseline_realized_objective_value": baseline_realized.objective_value,
        "baseline_realized_covered_demand": baseline_realized.covered_demand,
        "baseline_regret_vs_oracle": oracle_objective
        - baseline_realized.objective_value,
        "full_predicted_objective_value": full_predicted.objective_value,
        "full_predicted_covered_demand": full_predicted.covered_demand,
        "full_realized_objective_value": full_realized.objective_value,
        "full_realized_covered_demand": full_realized.covered_demand,
        "full_regret_vs_oracle": oracle_objective - full_realized.objective_value,
        "full_minus_baseline_realized_objective": full_realized.objective_value
        - baseline_realized.objective_value,
        "baseline_selected_zip_codes": json.dumps(
            list(baseline_predicted.selected_zip_codes)
        ),
        "full_selected_zip_codes": json.dumps(list(full_predicted.selected_zip_codes)),
        "oracle_selected_zip_codes": oracle_row["oracle_selected_zip_codes"],
        "baseline_solver_success": baseline_predicted.solver_success,
        "full_solver_success": full_predicted.solver_success,
        "oracle_solver_success": bool(oracle_row["oracle_solver_success"]),
    }


def _model_record_columns(model_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "xgb_n_estimators": int(model_record["n_estimators"]),
        "xgb_max_depth": int(model_record["max_depth"]),
        "xgb_learning_rate": float(model_record["learning_rate"]),
        "xgb_subsample": float(model_record["subsample"]),
        "xgb_colsample_bytree": float(model_record["colsample_bytree"]),
        "xgb_reg_lambda": float(model_record["reg_lambda"]),
    }


def _build_summary(per_hour: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "model_id",
        "regime",
        "tau",
        "p",
        "xgb_n_estimators",
        "xgb_max_depth",
        "xgb_learning_rate",
        "xgb_subsample",
        "xgb_colsample_bytree",
        "xgb_reg_lambda",
    ]
    summary = (
        per_hour.groupby(group_columns, as_index=False)
        .agg(
            evaluated_hours=("timestamp_hour", "count"),
            oracle_objective_mean=("oracle_objective_value", "mean"),
            baseline_objective_mean=("baseline_realized_objective_value", "mean"),
            full_objective_mean=("full_realized_objective_value", "mean"),
            baseline_regret_mean=("baseline_regret_vs_oracle", "mean"),
            full_regret_mean=("full_regret_vs_oracle", "mean"),
            full_minus_baseline_objective_mean=(
                "full_minus_baseline_realized_objective",
                "mean",
            ),
            oracle_covered_demand_mean=("oracle_covered_demand", "mean"),
            baseline_covered_demand_mean=("baseline_realized_covered_demand", "mean"),
            full_covered_demand_mean=("full_realized_covered_demand", "mean"),
            full_better_than_baseline_share=(
                "full_minus_baseline_realized_objective",
                lambda series: float((series > 0.0).mean()),
            ),
            full_worse_than_baseline_share=(
                "full_minus_baseline_realized_objective",
                lambda series: float((series < 0.0).mean()),
            ),
        )
        .sort_values(["regime", "model_id"])
        .reset_index(drop=True)
    )
    return summary


def _write_metadata(
    outdir: Path,
    *,
    args: argparse.Namespace,
    regimes: Sequence[Regime],
    model_ids: Sequence[str],
    prepared: dict[str, Any],
    runtime_seconds: float,
) -> None:
    time_split = prepared["time_split"]
    metadata = {
        "x_path": str(args.x_path),
        "y_path": str(args.y_path),
        "metadata_path": str(args.metadata_path),
        "zone_order_path": str(args.zone_order_path),
        "distance_matrix_path": str(args.distance_matrix_path),
        "model_ids": list(model_ids),
        "regimes": [
            {"regime": regime.label, "tau": regime.tau, "p": regime.p}
            for regime in regimes
        ],
        "random_state": int(args.random_state),
        "background_rows": int(args.background_rows),
        "test_months": int(args.test_months),
        "holdout_start": str(time_split.holdout_start),
        "holdout_end": str(time_split.holdout_end),
        "holdout_rows": int(len(time_split.holdout_x)),
        "sampled_eval_hours": int(len(prepared["explained_rows"])),
        "rest_holdout_hours": int(len(prepared["rest_x"])),
        "sampled_eval_rows": list(prepared["explained_rows"]),
        "solver": "scipy.optimize.milp / HiGHS",
        "objective_tolerance": float(args.objective_tolerance),
        "solver_time_limit_seconds": float(args.solver_time_limit_seconds),
        "runtime_seconds": float(runtime_seconds),
    }
    (outdir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
