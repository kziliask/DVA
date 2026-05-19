from __future__ import annotations

import argparse
import gc
import json
import math
import shlex
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dva.analysis.ems_exact_shap import (
    DEFAULT_BACKGROUND_ROWS,
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
    MaximumCoverageResult,
    _build_prediction_metrics,
    _build_solver_params,
    _build_time_split,
    _covered_demand_value,
    _fit_xgb_regressor,
    _frame_to_feature_matrix,
    _load_distance_matrix,
    _load_ems_frames,
    _load_zone_order,
    _maybe_sample_training_rows,
    _realized_coverage_value,
    _resolve_feature_columns,
    _sample_explanation_hours,
    _target_columns,
    _total_demand,
    _validate_config,
    build_coverage_matrix,
    compute_exact_shapley_values,
    normalize_ems_coverage_solver,
    solve_ems_coverage,
)
from dva.case_studies.ems.models import (
    EMS_XGB_MODEL_IDS,
    ems_xgb_config_kwargs,
    resolve_ems_xgb_model_record,
)


DEFAULT_OUTPUT_ROOT = Path("results/ems_design_utility_experiment")
DEFAULT_HOLDOUT_HOURS = 100
DEFAULT_TARGET_SOLVERS = (
    "exact",
    "greedy_max_cover",
    "lp_relaxation",
    "naive_greedy",
)
DEFAULT_COVERAGE_RADII_KM = (1.0, 2.0, 3.0)
DEFAULT_FACILITY_BUDGETS = (3, 5, 8)
DEFAULT_LAMBDAS = (0.0025, 0.005, 0.01, 0.02, 0.05)
DEFAULT_PRIMARY_LAMBDA = 0.01
DEFAULT_TIME_PENALTY_MIN_SECONDS = 0.0050
DEFAULT_TIME_PENALTY_MAX_SECONDS = 0.4056
DEFAULT_RUNTIME_EPSILON = 1e-6
DEFAULT_REPETITIONS = 3
DEFAULT_WARMUP_HOURS = 1
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 0

UTILITY_COVERAGE = "coverage"
UTILITY_NET = "net"
UTILITY_RUNTIME_LOG = "runtime_log"
UTILITY_RUNTIME_NORM = "runtime_norm"

_SOLVER_ALIASES = {
    "exact": "exact",
    "gurobi": "exact",
    "mip": "exact",
    "naive": "naive_greedy",
    "naive_greedy": "naive_greedy",
    "naive-greedy": "naive_greedy",
    "greedy": "greedy_max_cover",
    "greedy_max_cover": "greedy_max_cover",
    "greedy-max-cover": "greedy_max_cover",
    "lp": "lp_relaxation",
    "lp_relaxation": "lp_relaxation",
    "lp-relaxation": "lp_relaxation",
    "linear_relaxation": "lp_relaxation",
    "linear-relaxation": "lp_relaxation",
    "relaxation": "lp_relaxation",
}

_SOLVER_LABELS = {
    "exact": "exact",
    "naive_greedy": "naive",
    "greedy_max_cover": "greedy",
    "lp_relaxation": "lp_relaxation",
}


@dataclass(frozen=True, slots=True)
class EmsDesignSpec:
    solver: str
    radius_km: float
    facility_budget: int

    @property
    def design_id(self) -> str:
        return _design_id(self)


@dataclass(frozen=True, slots=True)
class DesignGameSpec:
    game_id: str
    reference: EmsDesignSpec
    target: EmsDesignSpec
    players: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedDesignUtilityData:
    config: EmsExactShapConfig
    model_record: Mapping[str, Any]
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    zip_codes: tuple[str, ...]
    explained_timestamps: tuple[str, ...]
    full_predictions: tuple[np.ndarray, ...]
    true_demands: tuple[np.ndarray, ...]
    prediction_metrics: dict[str, Any]
    preparation_metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EMS coverage, net-utility, and runtime-only DesignDVA over "
            "optimization design choices using full EMS predictions."
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
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-id", choices=EMS_XGB_MODEL_IDS, default="xgb_001")
    parser.add_argument("--reference-solver", default="exact")
    parser.add_argument(
        "--reference-coverage-radius-km",
        "--reference-tau",
        type=float,
        default=1.0,
    )
    parser.add_argument("--reference-facility-budget", "--reference-p", type=int, default=3)
    parser.add_argument(
        "--target-solver",
        action="append",
        default=None,
        help="Target solver to include. Repeat for multiple solvers.",
    )
    parser.add_argument(
        "--coverage-radius-km",
        "--tau",
        nargs="+",
        type=float,
        default=list(DEFAULT_COVERAGE_RADII_KM),
    )
    parser.add_argument(
        "--facility-budget",
        "--p",
        nargs="+",
        type=int,
        default=list(DEFAULT_FACILITY_BUDGETS),
    )
    parser.add_argument(
        "--lambda-value",
        "--lambda",
        dest="lambda_values",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
    )
    parser.add_argument("--primary-lambda", type=float, default=DEFAULT_PRIMARY_LAMBDA)
    parser.add_argument(
        "--time-penalty-min-seconds",
        type=float,
        default=DEFAULT_TIME_PENALTY_MIN_SECONDS,
    )
    parser.add_argument(
        "--time-penalty-max-seconds",
        type=float,
        default=DEFAULT_TIME_PENALTY_MAX_SECONDS,
    )
    parser.add_argument("--runtime-epsilon", type=float, default=DEFAULT_RUNTIME_EPSILON)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmup-hours", type=int, default=DEFAULT_WARMUP_HOURS)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--holdout-hours", type=int, default=DEFAULT_HOLDOUT_HOURS)
    parser.add_argument("--test-months", type=int, default=DEFAULT_TEST_MONTHS)
    parser.add_argument("--max-hours", type=int, default=None)
    parser.add_argument("--background-rows", type=int, default=DEFAULT_BACKGROUND_ROWS)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--solver-seed", type=int, default=None)
    parser.add_argument("--train-sample-rows", type=int, default=None)
    parser.add_argument("--xgb-verbosity", type=int, default=0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--mip-gap-abs", type=float, default=1e-9)
    parser.add_argument(
        "--solver-threads",
        "--gurobi-threads",
        dest="gurobi_threads",
        type=int,
        default=1,
    )
    parser.add_argument("--optimization-solver", choices=("highs", "gurobi"), default="highs")
    parser.add_argument(
        "--objective-tolerance",
        type=float,
        default=DEFAULT_OBJECTIVE_TOLERANCE,
    )
    parser.add_argument(
        "--exclude-zip-codes",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_ZIP_CODES),
    )
    parser.add_argument(
        "--no-gc-between-trials",
        dest="gc_between_trials",
        action="store_false",
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.dry_run:
        print(shlex.join(_dry_run_command(args)))
        return

    out_root = Path(args.out_root)
    if _outputs_are_complete(out_root) and not args.overwrite:
        print(f"Loading complete existing EMS design-utility outputs from {out_root}")
        return

    experiment_started_ns = time.perf_counter_ns()
    out_root.mkdir(parents=True, exist_ok=True)
    plot_dir = out_root / "plots"
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    reference = EmsDesignSpec(
        solver=_normalize_solver(args.reference_solver),
        radius_km=float(args.reference_coverage_radius_km),
        facility_budget=int(args.reference_facility_budget),
    )
    target_solvers = _resolve_solvers(args.target_solver)
    lambda_values = _resolve_lambda_values(args.lambda_values)
    if not any(np.isclose(float(args.primary_lambda), value) for value in lambda_values):
        lambda_values = (*lambda_values, float(args.primary_lambda))
    target_designs = _build_target_designs(
        target_solvers=target_solvers,
        coverage_radii_km=args.coverage_radius_km,
        facility_budgets=args.facility_budget,
    )
    design_games = _build_design_games(reference, target_designs)
    unique_designs = _unique_designs_for_games(reference, target_designs, design_games)
    _write_design_game_manifest(design_games, out_root / "design_game_manifest.csv")

    print(
        f"Preparing EMS model inputs for {args.model_id}; "
        f"{len(unique_designs)} unique design configuration(s), "
        f"{len(design_games)} DesignDVA game(s).",
        flush=True,
    )
    model_record = resolve_ems_xgb_model_record(args.model_id)
    solver_seed = args.random_state if args.solver_seed is None else args.solver_seed
    base_config = _build_base_config(args, model_record, reference, solver_seed)
    prepared = _prepare_design_utility_data(base_config, model_record)
    evaluation, raw_timing = _evaluate_design_configurations(
        prepared=prepared,
        designs=unique_designs,
        args=args,
        base_config=base_config,
    )
    evaluation = _add_utility_columns(
        evaluation,
        lambda_values=lambda_values,
        time_min_seconds=args.time_penalty_min_seconds,
        time_max_seconds=args.time_penalty_max_seconds,
        runtime_epsilon=args.runtime_epsilon,
    )
    coalition_values, hourly_dva = _build_design_dva_frames(
        design_games=design_games,
        evaluation=evaluation,
        lambda_values=lambda_values,
        model_id=args.model_id,
    )
    summary_dva = _summarize_hourly_design_dva(
        hourly_dva,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    configuration_summary = _build_configuration_summary(
        evaluation,
        lambda_values=lambda_values,
        primary_lambda=float(args.primary_lambda),
    )

    raw_timing.to_csv(out_root / "raw_solve_timing.csv", index=False)
    coalition_values.to_csv(out_root / "coalition_values.csv", index=False)
    hourly_dva.to_csv(out_root / "hourly_design_dva.csv", index=False)
    summary_dva.to_csv(out_root / "summary_design_dva.csv", index=False)
    configuration_summary.to_csv(out_root / "configuration_summary.csv", index=False)
    _write_configuration_markdown(
        configuration_summary,
        out_root / "configuration_summary.md",
        primary_lambda=float(args.primary_lambda),
    )

    if not args.no_plots:
        _write_plots(
            summary_dva=summary_dva,
            outdir=plot_dir,
            primary_lambda=float(args.primary_lambda),
        )

    metadata = {
        "experiment": "ems_design_utility_dva",
        "description": (
            "Coverage, net-utility, and runtime-only computational-value "
            "DesignDVA over EMS optimization design choices."
        ),
        "model_id": args.model_id,
        "model_record": dict(model_record),
        "reference_design": _design_dict(reference),
        "target_solvers": list(target_solvers),
        "coverage_radii_km": [float(value) for value in args.coverage_radius_km],
        "facility_budgets": [int(value) for value in args.facility_budget],
        "lambda_values": [float(value) for value in lambda_values],
        "primary_lambda": float(args.primary_lambda),
        "time_penalty_min_seconds": float(args.time_penalty_min_seconds),
        "time_penalty_max_seconds": float(args.time_penalty_max_seconds),
        "runtime_epsilon": float(args.runtime_epsilon),
        "repetitions": int(args.repetitions),
        "warmup_hours": int(args.warmup_hours),
        "bootstrap_draws": int(args.bootstrap_draws),
        "bootstrap_seed": int(args.bootstrap_seed),
        "optimization_solver": str(args.optimization_solver),
        "solver_threads": int(args.gurobi_threads),
        "prediction_time_excluded_from_utility": True,
        "utility_time_source": "median wall-clock optimization time",
        "zero_demand_coverage_convention": "0.0",
        "prepared_data": prepared.preparation_metadata,
        "prediction_metrics": prepared.prediction_metrics,
        "runtime_seconds": _elapsed_seconds(experiment_started_ns),
        "output_files": {
            "design_game_manifest": "design_game_manifest.csv",
            "coalition_values": "coalition_values.csv",
            "raw_solve_timing": "raw_solve_timing.csv",
            "hourly_design_dva": "hourly_design_dva.csv",
            "summary_design_dva": "summary_design_dva.csv",
            "configuration_summary": "configuration_summary.csv",
        },
    }
    _write_json(out_root / "experiment_metadata.json", metadata)
    print(f"Wrote EMS design-utility DesignDVA outputs to {out_root}", flush=True)


def normalized_log_time_penalty(
    solve_time_seconds: float,
    *,
    time_min_seconds: float = DEFAULT_TIME_PENALTY_MIN_SECONDS,
    time_max_seconds: float = DEFAULT_TIME_PENALTY_MAX_SECONDS,
) -> float:
    if solve_time_seconds < 0.0:
        raise ValueError("solve_time_seconds must be non-negative.")
    if time_min_seconds < 0.0 or time_max_seconds < 0.0:
        raise ValueError("time penalty bounds must be non-negative.")
    if time_max_seconds <= time_min_seconds:
        raise ValueError("time_max_seconds must be greater than time_min_seconds.")
    denominator = math.log1p(time_max_seconds) - math.log1p(time_min_seconds)
    return float((math.log1p(solve_time_seconds) - math.log1p(time_min_seconds)) / denominator)


def compute_design_utilities(
    *,
    coverage: float,
    solve_time_seconds: float,
    lambda_value: float,
    time_min_seconds: float = DEFAULT_TIME_PENALTY_MIN_SECONDS,
    time_max_seconds: float = DEFAULT_TIME_PENALTY_MAX_SECONDS,
    runtime_epsilon: float = DEFAULT_RUNTIME_EPSILON,
) -> dict[str, float]:
    if runtime_epsilon <= 0.0:
        raise ValueError("runtime_epsilon must be strictly positive.")
    penalty = normalized_log_time_penalty(
        solve_time_seconds,
        time_min_seconds=time_min_seconds,
        time_max_seconds=time_max_seconds,
    )
    return {
        "coverage_utility": float(coverage),
        "time_penalty": penalty,
        "net_utility": float(coverage) - float(lambda_value) * penalty,
        "runtime_log_utility": -math.log(float(runtime_epsilon) + float(solve_time_seconds)),
        "runtime_norm_utility": -penalty,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be strictly positive.")
    if args.warmup_hours < 0:
        raise ValueError("--warmup-hours must be non-negative.")
    if args.bootstrap_draws < 0:
        raise ValueError("--bootstrap-draws must be non-negative.")
    if args.holdout_hours <= 0:
        raise ValueError("--holdout-hours must be strictly positive.")
    if not args.coverage_radius_km:
        raise ValueError("At least one --coverage-radius-km value is required.")
    if not args.facility_budget:
        raise ValueError("At least one --facility-budget value is required.")
    if args.runtime_epsilon <= 0.0:
        raise ValueError("--runtime-epsilon must be strictly positive.")
    normalized_log_time_penalty(
        max(0.0, args.time_penalty_min_seconds),
        time_min_seconds=args.time_penalty_min_seconds,
        time_max_seconds=args.time_penalty_max_seconds,
    )


def _dry_run_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "dva-ems-design-utility-dva",
        "--model-id",
        args.model_id,
        "--out-root",
        str(args.out_root),
    ]
    for solver in args.target_solver or ():
        command.extend(["--target-solver", str(solver)])
    command.extend(
        [
            "--coverage-radius-km",
            *[str(value) for value in args.coverage_radius_km],
            "--facility-budget",
            *[str(value) for value in args.facility_budget],
            "--holdout-hours",
            str(args.holdout_hours),
            "--background-rows",
            str(args.background_rows),
            "--repetitions",
            str(args.repetitions),
            "--warmup-hours",
            str(args.warmup_hours),
            "--optimization-solver",
            str(args.optimization_solver),
            "--solver-threads",
            str(args.gurobi_threads),
        ]
    )
    if args.max_hours is not None:
        command.extend(["--max-hours", str(args.max_hours)])
    if args.overwrite:
        command.append("--overwrite")
    if args.no_plots:
        command.append("--no-plots")
    return command


def _build_base_config(
    args: argparse.Namespace,
    model_record: Mapping[str, Any],
    reference: EmsDesignSpec,
    solver_seed: int,
) -> EmsExactShapConfig:
    config = EmsExactShapConfig(
        x_path=args.x_path,
        y_path=args.y_path,
        metadata_path=args.metadata_path,
        zone_order_path=args.zone_order_path,
        distance_matrix_path=args.distance_matrix_path,
        outdir=args.out_root,
        holdout_hours=args.holdout_hours,
        test_months=args.test_months,
        max_hours=args.max_hours,
        background_rows=args.background_rows,
        random_state=args.random_state,
        model_id=args.model_id,
        **ems_xgb_config_kwargs(model_record),
        xgb_verbosity=args.xgb_verbosity,
        train_sample_rows=args.train_sample_rows,
        coverage_radius_km=reference.radius_km,
        facility_budget=reference.facility_budget,
        solver_seed=solver_seed,
        mip_gap=args.mip_gap,
        mip_gap_abs=args.mip_gap_abs,
        gurobi_threads=args.gurobi_threads,
        optimization_solver=args.optimization_solver,
        objective_tolerance=args.objective_tolerance,
        coverage_solver=reference.solver,
        excluded_zip_codes=tuple(str(zip_code) for zip_code in args.exclude_zip_codes),
        save_coalition_values=False,
        compute_cvar_decision_shap=False,
    )
    _validate_config(config)
    return config


def _prepare_design_utility_data(
    config: EmsExactShapConfig,
    model_record: Mapping[str, Any],
) -> PreparedDesignUtilityData:
    started_ns = time.perf_counter_ns()
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
    unsampled_train_rows = len(time_split.train_x)
    train_x, train_y = _maybe_sample_training_rows(
        time_split.train_x,
        time_split.train_y,
        train_sample_rows=config.train_sample_rows,
        random_state=config.random_state,
    )
    explain_x, explain_y, explained_rows = _sample_explanation_hours(
        time_split.holdout_x,
        time_split.holdout_y,
        time_split.holdout_source_rows,
        holdout_hours=config.holdout_hours,
        max_hours=config.max_hours,
        random_state=config.random_state,
    )
    if explain_x.empty:
        raise ValueError("No explanation hours were sampled from the EMS holdout set.")

    print(
        f"Training EMS XGBRegressor on {len(train_x):,} rows; "
        f"explaining {len(explain_x):,} holdout hour(s).",
        flush=True,
    )
    training_started_ns = time.perf_counter_ns()
    model = _fit_xgb_regressor(
        train_frame=train_x,
        y_train=train_y.loc[:, list(target_columns)],
        feature_columns=feature_columns,
        config=config,
    )
    training_seconds = _elapsed_seconds(training_started_ns)

    holdout_predictions = np.maximum(
        np.asarray(
            model.predict(_frame_to_feature_matrix(time_split.holdout_x, feature_columns)),
            dtype=float,
        ),
        0.0,
    )
    if holdout_predictions.ndim == 1:
        holdout_predictions = holdout_predictions[:, np.newaxis]
    prediction_metrics = _build_prediction_metrics(
        y_true=time_split.holdout_y.loc[:, list(target_columns)].to_numpy(
            dtype=float,
            copy=True,
        ),
        y_pred=holdout_predictions,
    )

    full_predictions = np.maximum(
        np.asarray(
            model.predict(_frame_to_feature_matrix(explain_x, feature_columns)),
            dtype=float,
        ),
        0.0,
    )
    if full_predictions.ndim == 1:
        full_predictions = full_predictions[:, np.newaxis]

    true_demands = tuple(
        explain_y.loc[hour_idx, list(target_columns)].to_numpy(dtype=float, copy=True)
        for hour_idx in range(len(explain_y))
    )
    explained_timestamps = tuple(str(timestamp) for timestamp in explain_x[EMS_TIMESTAMP_COLUMN])
    preparation_metadata = {
        "candidate_train_rows": int(unsampled_train_rows),
        "train_rows": int(len(train_x)),
        "holdout_rows": int(len(time_split.holdout_x)),
        "train_start": str(time_split.train_start),
        "train_end": str(time_split.train_end),
        "holdout_start": str(time_split.holdout_start),
        "holdout_end": str(time_split.holdout_end),
        "explained_rows": list(explained_rows),
        "explained_hours": list(explained_timestamps),
        "training_seconds": training_seconds,
        "preparation_seconds": _elapsed_seconds(started_ns),
        "target_count": int(len(target_columns)),
        "zip_count": int(len(zip_codes)),
    }
    return PreparedDesignUtilityData(
        config=config,
        model_record=model_record,
        feature_columns=tuple(feature_columns),
        target_columns=target_columns,
        zip_codes=zip_codes,
        explained_timestamps=explained_timestamps,
        full_predictions=tuple(np.asarray(row, dtype=float) for row in full_predictions),
        true_demands=true_demands,
        prediction_metrics=prediction_metrics,
        preparation_metadata=preparation_metadata,
    )


def _evaluate_design_configurations(
    *,
    prepared: PreparedDesignUtilityData,
    designs: Sequence[EmsDesignSpec],
    args: argparse.Namespace,
    base_config: EmsExactShapConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    distance_matrix = _load_distance_matrix(args.distance_matrix_path)
    coverage_by_radius = {
        float(radius): build_coverage_matrix(
            distance_matrix,
            prepared.zip_codes,
            coverage_radius_km=float(radius),
        )
        for radius in sorted({design.radius_km for design in designs})
    }
    config_by_design = {
        design.design_id: replace(
            base_config,
            coverage_solver=design.solver,
            coverage_radius_km=design.radius_km,
            facility_budget=design.facility_budget,
        )
        for design in designs
    }

    if args.warmup_hours > 0:
        print(
            f"Running {args.warmup_hours} warmup hour(s) for each design configuration.",
            flush=True,
        )
        warmup_count = min(int(args.warmup_hours), len(prepared.explained_timestamps))
        for design in designs:
            config = config_by_design[design.design_id]
            coverage_matrix = coverage_by_radius[design.radius_km]
            for hour_idx in range(warmup_count):
                solve_ems_coverage(
                    prepared.full_predictions[hour_idx],
                    coverage_matrix,
                    prepared.zip_codes,
                    facility_budget=design.facility_budget,
                    solver_name=design.solver,
                    name=f"ems_design_utility_warmup_{design.design_id}_{hour_idx}",
                    solver_params=_build_solver_params(config),
                    optimization_solver=config.optimization_solver,
                    objective_tolerance=config.objective_tolerance,
                )

    evaluation_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    total_trials = len(designs) * len(prepared.explained_timestamps) * int(args.repetitions)
    completed_trials = 0
    report_every = max(1, total_trials // 20)
    for design in designs:
        config = config_by_design[design.design_id]
        coverage_matrix = coverage_by_radius[design.radius_km]
        for hour_idx, timestamp_hour in enumerate(prepared.explained_timestamps):
            solutions: list[MaximumCoverageResult] = []
            wall_times: list[float] = []
            solver_reported_times: list[float] = []
            for repetition in range(int(args.repetitions)):
                if args.gc_between_trials:
                    gc.collect()
                solve_started_ns = time.perf_counter_ns()
                solution = solve_ems_coverage(
                    prepared.full_predictions[hour_idx],
                    coverage_matrix,
                    prepared.zip_codes,
                    facility_budget=design.facility_budget,
                    solver_name=design.solver,
                    name=(
                        f"ems_design_utility_{design.design_id}_"
                        f"h{hour_idx}_r{repetition}"
                    ),
                    solver_params=_build_solver_params(config),
                    optimization_solver=config.optimization_solver,
                    objective_tolerance=config.objective_tolerance,
                )
                wall_seconds = _elapsed_seconds(solve_started_ns)
                solutions.append(solution)
                wall_times.append(wall_seconds)
                if solution.solver_runtime_seconds is not None:
                    solver_reported_times.append(float(solution.solver_runtime_seconds))
                timing_rows.append(
                    {
                        "model_id": prepared.config.model_id,
                        "design_id": design.design_id,
                        "coverage_solver": design.solver,
                        "coverage_solver_label": _solver_label(design.solver),
                        "coverage_radius_km": float(design.radius_km),
                        "facility_budget": int(design.facility_budget),
                        "hour_index": int(hour_idx),
                        "timestamp_hour": timestamp_hour,
                        "repetition": int(repetition),
                        "wall_clock_solve_seconds": wall_seconds,
                        "solver_reported_runtime_seconds": solution.solver_runtime_seconds,
                        "selected_facility_zip_codes": json.dumps(
                            list(solution.selected_facility_zip_codes)
                        ),
                        "covered_zip_codes": json.dumps(list(solution.covered_zip_codes)),
                        "solver_status": solution.solver_status,
                        "optimal": bool(solution.optimal),
                        "mip_gap": solution.mip_gap,
                    }
                )
                completed_trials += 1
                if completed_trials == total_trials or completed_trials % report_every == 0:
                    print(
                        f"Completed {completed_trials:,}/{total_trials:,} timed "
                        "EMS design solve trial(s).",
                        flush=True,
                    )
            median_index = int(np.argsort(np.asarray(wall_times, dtype=float))[len(wall_times) // 2])
            median_solution = solutions[median_index]
            true_demand = prepared.true_demands[hour_idx]
            actual_total_demand = _total_demand(true_demand)
            realized_covered_demand = _covered_demand_value(
                median_solution.covered_zone_indices,
                true_demand,
            )
            evaluation_rows.append(
                {
                    "model_id": prepared.config.model_id,
                    "design_id": design.design_id,
                    "coverage_solver": design.solver,
                    "coverage_solver_label": _solver_label(design.solver),
                    "coverage_radius_km": float(design.radius_km),
                    "facility_budget": int(design.facility_budget),
                    "hour_index": int(hour_idx),
                    "timestamp_hour": timestamp_hour,
                    "predicted_total_demand": float(_total_demand(prepared.full_predictions[hour_idx])),
                    "actual_total_demand": actual_total_demand,
                    "realized_covered_demand": realized_covered_demand,
                    "realized_coverage": _realized_coverage_value(
                        median_solution.covered_zone_indices,
                        true_demand,
                    ),
                    "median_wall_clock_solve_seconds": float(np.median(wall_times)),
                    "mean_wall_clock_solve_seconds": float(np.mean(wall_times)),
                    "min_wall_clock_solve_seconds": float(np.min(wall_times)),
                    "max_wall_clock_solve_seconds": float(np.max(wall_times)),
                    "median_solver_reported_runtime_seconds": (
                        None
                        if not solver_reported_times
                        else float(np.median(solver_reported_times))
                    ),
                    "selected_facility_zip_codes": json.dumps(
                        list(median_solution.selected_facility_zip_codes)
                    ),
                    "covered_zip_codes": json.dumps(list(median_solution.covered_zip_codes)),
                    "selected_facility_count": len(median_solution.selected_facility_zip_codes),
                    "covered_zone_count": len(median_solution.covered_zip_codes),
                    "solver_status": median_solution.solver_status,
                    "optimal": bool(median_solution.optimal),
                    "mip_gap": median_solution.mip_gap,
                }
            )
    return pd.DataFrame(evaluation_rows), pd.DataFrame(timing_rows)


def _add_utility_columns(
    evaluation: pd.DataFrame,
    *,
    lambda_values: Sequence[float],
    time_min_seconds: float,
    time_max_seconds: float,
    runtime_epsilon: float,
) -> pd.DataFrame:
    frame = evaluation.copy()
    frame["time_penalty"] = [
        normalized_log_time_penalty(
            float(value),
            time_min_seconds=time_min_seconds,
            time_max_seconds=time_max_seconds,
        )
        for value in frame["median_wall_clock_solve_seconds"]
    ]
    frame["coverage_utility"] = frame["realized_coverage"].astype(float)
    frame["runtime_log_utility"] = [
        -math.log(float(runtime_epsilon) + float(value))
        for value in frame["median_wall_clock_solve_seconds"]
    ]
    frame["runtime_norm_utility"] = -frame["time_penalty"].astype(float)
    for lambda_value in lambda_values:
        column = _net_utility_column(lambda_value)
        frame[column] = frame["coverage_utility"] - float(lambda_value) * frame["time_penalty"]
    return frame


def _build_design_dva_frames(
    *,
    design_games: Sequence[DesignGameSpec],
    evaluation: pd.DataFrame,
    lambda_values: Sequence[float],
    model_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_by_key = {
        (str(row.timestamp_hour), str(row.design_id)): row._asdict()
        for row in evaluation.itertuples(index=False)
    }
    timestamps = tuple(dict.fromkeys(evaluation["timestamp_hour"].astype(str)))
    coalition_rows: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []
    for game in design_games:
        for timestamp_hour in timestamps:
            coalition_by_mask: dict[int, dict[str, Any]] = {}
            for mask in range(1 << len(game.players)):
                design = design_for_mask(mask, game=game)
                eval_row = eval_by_key[(timestamp_hour, design.design_id)]
                coalition_by_mask[mask] = eval_row
                coalition_rows.append(
                    {
                        "game_id": game.game_id,
                        "coalition_mask": int(mask),
                        "coalition_bits": f"{mask:0{len(game.players)}b}",
                        "timestamp_hour": timestamp_hour,
                        "model_id": model_id,
                        "design_id": design.design_id,
                        "coverage_solver": design.solver,
                        "coverage_solver_label": _solver_label(design.solver),
                        "coverage_radius_km": float(design.radius_km),
                        "facility_budget": int(design.facility_budget),
                        "target_design_id": game.target.design_id,
                        "reference_design_id": game.reference.design_id,
                        "design_players": "|".join(game.players),
                        "realized_coverage": float(eval_row["realized_coverage"]),
                        "realized_covered_demand": float(eval_row["realized_covered_demand"]),
                        "actual_total_demand": float(eval_row["actual_total_demand"]),
                        "median_wall_clock_solve_seconds": float(
                            eval_row["median_wall_clock_solve_seconds"]
                        ),
                        "median_solver_reported_runtime_seconds": eval_row[
                            "median_solver_reported_runtime_seconds"
                        ],
                        "time_penalty": float(eval_row["time_penalty"]),
                        "coverage_utility": float(eval_row["coverage_utility"]),
                        "runtime_log_utility": float(eval_row["runtime_log_utility"]),
                        "runtime_norm_utility": float(eval_row["runtime_norm_utility"]),
                        **{
                            _net_utility_column(lambda_value): float(
                                eval_row[_net_utility_column(lambda_value)]
                            )
                            for lambda_value in lambda_values
                        },
                        "selected_facility_zip_codes": eval_row[
                            "selected_facility_zip_codes"
                        ],
                        "covered_zip_codes": eval_row["covered_zip_codes"],
                    }
                )
            _append_hourly_shapley_rows(
                hourly_rows,
                game=game,
                timestamp_hour=timestamp_hour,
                model_id=model_id,
                utility_kind=UTILITY_COVERAGE,
                lambda_value=None,
                values=np.asarray(
                    [
                        coalition_by_mask[mask]["coverage_utility"]
                        for mask in range(1 << len(game.players))
                    ],
                    dtype=float,
                ),
            )
            _append_hourly_shapley_rows(
                hourly_rows,
                game=game,
                timestamp_hour=timestamp_hour,
                model_id=model_id,
                utility_kind=UTILITY_RUNTIME_LOG,
                lambda_value=None,
                values=np.asarray(
                    [
                        coalition_by_mask[mask]["runtime_log_utility"]
                        for mask in range(1 << len(game.players))
                    ],
                    dtype=float,
                ),
            )
            _append_hourly_shapley_rows(
                hourly_rows,
                game=game,
                timestamp_hour=timestamp_hour,
                model_id=model_id,
                utility_kind=UTILITY_RUNTIME_NORM,
                lambda_value=None,
                values=np.asarray(
                    [
                        coalition_by_mask[mask]["runtime_norm_utility"]
                        for mask in range(1 << len(game.players))
                    ],
                    dtype=float,
                ),
            )
            for lambda_value in lambda_values:
                _append_hourly_shapley_rows(
                    hourly_rows,
                    game=game,
                    timestamp_hour=timestamp_hour,
                    model_id=model_id,
                    utility_kind=UTILITY_NET,
                    lambda_value=float(lambda_value),
                    values=np.asarray(
                        [
                            coalition_by_mask[mask][_net_utility_column(lambda_value)]
                            for mask in range(1 << len(game.players))
                        ],
                        dtype=float,
                    ),
                )
    return pd.DataFrame(coalition_rows), pd.DataFrame(hourly_rows)


def _append_hourly_shapley_rows(
    rows: list[dict[str, Any]],
    *,
    game: DesignGameSpec,
    timestamp_hour: str,
    model_id: str,
    utility_kind: str,
    lambda_value: float | None,
    values: np.ndarray,
) -> None:
    characteristic_values = np.asarray(values, dtype=float) - float(values[0])
    shap_values = compute_exact_shapley_values(characteristic_values, len(game.players))
    for player, shap_value in zip(game.players, shap_values, strict=True):
        rows.append(
            {
                "game_id": game.game_id,
                "timestamp_hour": timestamp_hour,
                "model_id": model_id,
                "utility_kind": utility_kind,
                "lambda_value": lambda_value,
                "player": player,
                "baseline": getattr(game.reference, _player_attr(player)),
                "target": getattr(game.target, _player_attr(player)),
                "dva_value": float(shap_value),
                "reference_utility": float(values[0]),
                "target_utility": float(values[-1]),
                "utility_gain": float(characteristic_values[-1]),
                "shapley_sum": float(np.sum(shap_values)),
                "shapley_additivity_abs_error": abs(
                    float(np.sum(shap_values)) - float(characteristic_values[-1])
                ),
                "reference_design_id": game.reference.design_id,
                "target_design_id": game.target.design_id,
                "reference_solver": game.reference.solver,
                "target_solver": game.target.solver,
                "reference_coverage_radius_km": float(game.reference.radius_km),
                "target_coverage_radius_km": float(game.target.radius_km),
                "reference_facility_budget": int(game.reference.facility_budget),
                "target_facility_budget": int(game.target.facility_budget),
            }
        )


def _summarize_hourly_design_dva(
    hourly_dva: pd.DataFrame,
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if hourly_dva.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(bootstrap_seed)
    group_columns = [
        "game_id",
        "utility_kind",
        "lambda_value",
        "player",
        "reference_design_id",
        "target_design_id",
    ]
    rows: list[dict[str, Any]] = []
    grouped = hourly_dva.groupby(group_columns, dropna=False, sort=True)
    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {
            column: value
            for column, value in zip(group_columns, group_values, strict=True)
        }
        values = group["dva_value"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(values, rng, bootstrap_draws)
        row.update(
            {
                "n_hours": int(len(values)),
                "mean_dva_value": float(np.mean(values)),
                "mean_dva_value_ci95_low": ci_low,
                "mean_dva_value_ci95_high": ci_high,
                "mean_abs_dva_value": float(np.mean(np.abs(values))),
                "median_dva_value": float(np.median(values)),
                "q25_dva_value": float(np.quantile(values, 0.25)),
                "q75_dva_value": float(np.quantile(values, 0.75)),
                "max_shapley_additivity_abs_error": float(
                    group["shapley_additivity_abs_error"].max()
                ),
            }
        )
        for metadata_column in (
            "baseline",
            "target",
            "reference_solver",
            "target_solver",
            "reference_coverage_radius_km",
            "target_coverage_radius_km",
            "reference_facility_budget",
            "target_facility_budget",
        ):
            unique_values = group[metadata_column].drop_duplicates()
            if len(unique_values) == 1:
                row[metadata_column] = unique_values.iloc[0]
        rows.append(row)

    summary = pd.DataFrame(rows)
    rank_columns = ["game_id", "utility_kind", "lambda_value"]
    summary["dva_rank"] = (
        summary.groupby(rank_columns, dropna=False)["mean_abs_dva_value"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return summary.sort_values(
        ["utility_kind", "lambda_value", "game_id", "dva_rank"],
        na_position="first",
    ).reset_index(drop=True)


def _build_configuration_summary(
    evaluation: pd.DataFrame,
    *,
    lambda_values: Sequence[float],
    primary_lambda: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "design_id",
        "coverage_solver",
        "coverage_solver_label",
        "coverage_radius_km",
        "facility_budget",
    ]
    for group_values, group in evaluation.groupby(group_columns, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {
            column: value
            for column, value in zip(group_columns, group_values, strict=True)
        }
        row.update(
            {
                "hour_count": int(len(group)),
                "mean_realized_coverage": float(group["realized_coverage"].mean()),
                "mean_realized_covered_demand": float(
                    group["realized_covered_demand"].mean()
                ),
                "mean_actual_total_demand": float(group["actual_total_demand"].mean()),
                "mean_solve_time_seconds": float(
                    group["median_wall_clock_solve_seconds"].mean()
                ),
                "median_solve_time_seconds": float(
                    group["median_wall_clock_solve_seconds"].median()
                ),
                "mean_normalized_time_penalty": float(group["time_penalty"].mean()),
                "mean_runtime_log_utility": float(group["runtime_log_utility"].mean()),
                "mean_runtime_norm_utility": float(group["runtime_norm_utility"].mean()),
            }
        )
        for lambda_value in lambda_values:
            row[f"mean_{_net_utility_column(lambda_value)}"] = float(
                group[_net_utility_column(lambda_value)].mean()
            )
        row["mean_net_utility"] = row[f"mean_{_net_utility_column(primary_lambda)}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["coverage_solver_label", "coverage_radius_km", "facility_budget"]
    ).reset_index(drop=True)


def _build_target_designs(
    *,
    target_solvers: Sequence[str],
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
) -> tuple[EmsDesignSpec, ...]:
    designs = [
        EmsDesignSpec(
            solver=solver,
            radius_km=float(radius_km),
            facility_budget=int(facility_budget),
        )
        for solver in target_solvers
        for radius_km in coverage_radii_km
        for facility_budget in facility_budgets
    ]
    return tuple(dict.fromkeys(designs))


def _build_design_games(
    reference: EmsDesignSpec,
    target_designs: Sequence[EmsDesignSpec],
) -> tuple[DesignGameSpec, ...]:
    games: list[DesignGameSpec] = []
    for target in target_designs:
        players = design_players_for_designs(reference, target)
        if not players:
            continue
        game_id = f"{reference.design_id}_to_{target.design_id}"
        games.append(
            DesignGameSpec(
                game_id=game_id,
                reference=reference,
                target=target,
                players=players,
            )
        )
    return tuple(games)


def _unique_designs_for_games(
    reference: EmsDesignSpec,
    target_designs: Sequence[EmsDesignSpec],
    design_games: Sequence[DesignGameSpec],
) -> tuple[EmsDesignSpec, ...]:
    designs = {reference.design_id: reference}
    for target in target_designs:
        designs[target.design_id] = target
    for game in design_games:
        for mask in range(1 << len(game.players)):
            design = design_for_mask(mask, game=game)
            designs[design.design_id] = design
    return tuple(sorted(designs.values(), key=lambda design: design.design_id))


def design_players_for_designs(
    reference: EmsDesignSpec,
    target: EmsDesignSpec,
) -> tuple[str, ...]:
    players: list[str] = []
    if reference.solver != target.solver:
        players.append("solver")
    if not np.isclose(reference.radius_km, target.radius_km):
        players.append("radius_km")
    if int(reference.facility_budget) != int(target.facility_budget):
        players.append("facility_budget")
    return tuple(players)


def design_for_mask(mask: int, *, game: DesignGameSpec) -> EmsDesignSpec:
    values = _design_dict(game.reference)
    for player_idx, player in enumerate(game.players):
        if mask & (1 << player_idx):
            values[player] = getattr(game.target, _player_attr(player))
    return EmsDesignSpec(
        solver=str(values["solver"]),
        radius_km=float(values["radius_km"]),
        facility_budget=int(values["facility_budget"]),
    )


def _write_design_game_manifest(
    games: Sequence[DesignGameSpec],
    path: Path,
) -> None:
    rows = []
    for game in games:
        rows.append(
            {
                "game_id": game.game_id,
                "design_players": "|".join(game.players),
                "player_count": int(len(game.players)),
                "coalition_count": int(1 << len(game.players)),
                "reference_design_id": game.reference.design_id,
                "target_design_id": game.target.design_id,
                "reference_solver": game.reference.solver,
                "target_solver": game.target.solver,
                "reference_coverage_radius_km": float(game.reference.radius_km),
                "target_coverage_radius_km": float(game.target.radius_km),
                "reference_facility_budget": int(game.reference.facility_budget),
                "target_facility_budget": int(game.target.facility_budget),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_plots(
    *,
    summary_dva: pd.DataFrame,
    outdir: Path,
    primary_lambda: float,
) -> None:
    aggregate = _aggregate_summary_for_plots(summary_dva)
    _write_bar_plot(
        aggregate.loc[aggregate["utility_kind"].eq(UTILITY_COVERAGE)],
        outdir / "coverage_only_design_dva.png",
        title="Coverage-only DesignDVA",
        xlabel="Mean Shapley value",
    )
    net_primary = aggregate.loc[
        aggregate["utility_kind"].eq(UTILITY_NET)
        & np.isclose(aggregate["lambda_value"].astype(float), float(primary_lambda))
    ]
    _write_bar_plot(
        net_primary,
        outdir / "net_utility_design_dva_lambda_0p01.png",
        title=f"Net-utility DesignDVA, lambda={primary_lambda:g}",
        xlabel="Mean Shapley value",
    )
    _write_lambda_sensitivity_plot(
        aggregate.loc[aggregate["utility_kind"].eq(UTILITY_NET)],
        outdir / "lambda_sensitivity_design_dva.png",
    )


def _aggregate_summary_for_plots(summary_dva: pd.DataFrame) -> pd.DataFrame:
    if summary_dva.empty:
        return pd.DataFrame(columns=["utility_kind", "lambda_value", "player", "mean_dva_value"])
    group_columns = ["utility_kind", "lambda_value", "player"]
    rows: list[dict[str, Any]] = []
    for group_values, group in summary_dva.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {
            column: value
            for column, value in zip(group_columns, group_values, strict=True)
        }
        row["mean_dva_value"] = float(group["mean_dva_value"].mean())
        row["mean_abs_dva_value"] = float(group["mean_abs_dva_value"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_bar_plot(
    frame: pd.DataFrame,
    path: Path,
    *,
    title: str,
    xlabel: str,
) -> None:
    plot_frame = frame.dropna(subset=["mean_dva_value"]).copy()
    if plot_frame.empty:
        return
    plot_frame = plot_frame.sort_values("mean_dva_value")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.barh(plot_frame["player"], plot_frame["mean_dva_value"], color="#3b6ea8")
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Design player")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_lambda_sensitivity_plot(frame: pd.DataFrame, path: Path) -> None:
    plot_frame = frame.dropna(subset=["lambda_value", "mean_dva_value"]).copy()
    if plot_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for player, group in plot_frame.groupby("player", sort=True):
        group = group.sort_values("lambda_value")
        ax.plot(
            group["lambda_value"].astype(float),
            group["mean_dva_value"].astype(float),
            marker="o",
            label=str(player),
        )
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Mean net-utility Shapley value")
    ax.set_title("EMS net-utility DesignDVA sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_configuration_markdown(
    summary: pd.DataFrame,
    path: Path,
    *,
    primary_lambda: float,
) -> None:
    columns = [
        "coverage_solver_label",
        "coverage_radius_km",
        "facility_budget",
        "mean_realized_coverage",
        "mean_solve_time_seconds",
        "mean_normalized_time_penalty",
        "mean_net_utility",
    ]
    header = [
        "solver",
        "tau",
        "p",
        "coverage",
        "solve_s",
        "penalty",
        f"net_lambda_{primary_lambda:g}",
    ]
    lines = [
        "# EMS Design Utility Configuration Summary",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in summary.loc[:, columns].itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.coverage_solver_label),
                    f"{float(row.coverage_radius_km):g}",
                    str(int(row.facility_budget)),
                    f"{float(row.mean_realized_coverage):.6f}",
                    f"{float(row.mean_solve_time_seconds):.6f}",
                    f"{float(row.mean_normalized_time_penalty):.6f}",
                    f"{float(row.mean_net_utility):.6f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float]:
    cleaned = np.asarray(values, dtype=float)
    if cleaned.size == 0:
        raise ValueError("Cannot bootstrap an empty value array.")
    if draws <= 0 or cleaned.size <= 1:
        mean_value = float(np.mean(cleaned))
        return mean_value, mean_value
    sample_indices = rng.integers(0, cleaned.size, size=(int(draws), cleaned.size))
    means = cleaned[sample_indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _resolve_solvers(raw_solvers: Sequence[str] | None) -> tuple[str, ...]:
    if raw_solvers is None:
        return DEFAULT_TARGET_SOLVERS
    solvers = [_normalize_solver(solver_name) for solver_name in raw_solvers]
    return tuple(dict.fromkeys(solvers))


def _normalize_solver(solver_name: str) -> str:
    key = str(solver_name).strip().lower().replace(" ", "_")
    normalized = _SOLVER_ALIASES.get(key, key)
    return normalize_ems_coverage_solver(normalized)


def _resolve_lambda_values(raw_values: Sequence[float]) -> tuple[float, ...]:
    if not raw_values:
        raise ValueError("At least one lambda value is required.")
    values: list[float] = []
    for raw_value in raw_values:
        value = float(raw_value)
        if value < 0.0:
            raise ValueError("lambda values must be non-negative.")
        if not any(np.isclose(value, existing) for existing in values):
            values.append(value)
    return tuple(values)


def _design_dict(design: EmsDesignSpec) -> dict[str, Any]:
    return {
        "solver": design.solver,
        "radius_km": float(design.radius_km),
        "facility_budget": int(design.facility_budget),
    }


def _design_id(design: EmsDesignSpec) -> str:
    return (
        f"{_solver_label(design.solver)}_"
        f"tau{_format_numeric_id(design.radius_km)}_"
        f"p{int(design.facility_budget)}"
    )


def _player_attr(player: str) -> str:
    if player == "facility_budget":
        return "facility_budget"
    if player == "radius_km":
        return "radius_km"
    if player == "solver":
        return "solver"
    raise KeyError(f"Unknown EMS design player: {player}")


def _solver_label(solver: str) -> str:
    return _SOLVER_LABELS.get(str(solver), str(solver))


def _net_utility_column(lambda_value: float) -> str:
    return f"net_utility_lambda_{_format_numeric_id(float(lambda_value))}"


def _format_numeric_id(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _elapsed_seconds(started_ns: int) -> float:
    return float((time.perf_counter_ns() - started_ns) / 1_000_000_000.0)


def _outputs_are_complete(out_root: Path) -> bool:
    required_files = [
        "design_game_manifest.csv",
        "coalition_values.csv",
        "raw_solve_timing.csv",
        "hourly_design_dva.csv",
        "summary_design_dva.csv",
        "configuration_summary.csv",
        "experiment_metadata.json",
    ]
    return all((out_root / filename).exists() for filename in required_files)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
