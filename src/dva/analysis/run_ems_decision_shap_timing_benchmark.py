from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dva.analysis.ems_exact_shap import (
    DEFAULT_BACKGROUND_ROWS,
    DEFAULT_COALITION_BATCH_SIZE,
    DEFAULT_DISTANCE_MATRIX_PATH,
    DEFAULT_EXCLUDED_ZIP_CODES,
    DEFAULT_HOLDOUT_HOURS,
    DEFAULT_METADATA_PATH,
    DEFAULT_OBJECTIVE_TOLERANCE,
    DEFAULT_PROGRESS_EVERY_COALITIONS,
    DEFAULT_TEST_MONTHS,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_ZONE_ORDER_PATH,
    EMS_TIMESTAMP_COLUMN,
    EmsExactShapConfig,
    GroupedBackgroundCoalitionPredictor,
    _build_prediction_metrics,
    _build_time_split,
    _fit_xgb_regressor,
    _frame_to_feature_matrix,
    _load_distance_matrix,
    _load_ems_frames,
    _load_zone_order,
    _maybe_sample_training_rows,
    _resolve_feature_columns,
    _sample_background_frame,
    _sample_explanation_hours,
    _solve_decision_values,
    _target_columns,
    _validate_config,
    build_coverage_matrix,
    build_ems_feature_groups,
    compute_exact_shapley_values,
    normalize_ems_coverage_solver,
)


DEFAULT_OUTPUT_DIR = Path("results/ems_decision_shap_timing_benchmark")
DEFAULT_SOLVERS = (
    "naive_greedy",
    "greedy_max_cover",
    "gurobi_lp_relaxation",
    "gurobi",
)
DEFAULT_COVERAGE_RADII_KM = (2.0,)
DEFAULT_FACILITY_BUDGETS = (5,)
DEFAULT_REPETITIONS = 3
DEFAULT_WARMUP_HOURS = 1
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BENCHMARK_SEED = 20_260_510
_SOLVER_LABELS = {
    "naive_greedy": "naive",
    "greedy_max_cover": "greedy",
    "gurobi_lp_relaxation": "lp_relaxation",
    "gurobi": "exact",
}
_SOLVER_ALIASES = {
    "exact": "gurobi",
    "naive": "naive_greedy",
    "greedy": "greedy_max_cover",
    "linear_relaxation": "gurobi_lp_relaxation",
    "lp": "gurobi_lp_relaxation",
    "lp_relaxation": "gurobi_lp_relaxation",
    "relaxation": "gurobi_lp_relaxation",
}


@dataclass(frozen=True, slots=True)
class BenchmarkSetting:
    setting_id: str
    coverage_radius_km: float
    facility_budget: int
    coverage_matrix: np.ndarray
    coverage_matrix_density: float


@dataclass(frozen=True, slots=True)
class BenchmarkPreparedData:
    config: EmsExactShapConfig
    feature_columns: tuple[str, ...]
    player_names: tuple[str, ...]
    target_columns: tuple[str, ...]
    zip_codes: tuple[str, ...]
    explained_timestamps: tuple[str, ...]
    true_demands: tuple[np.ndarray, ...]
    coalition_predictions: tuple[np.ndarray, ...]
    predictive_timing: pd.DataFrame
    prediction_metrics: dict[str, Any]
    preparation_metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark EMS decision-SHAP attribution compute time for the four "
            "maximum-coverage methods used in the exhaustive comparison."
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
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--solver",
        action="append",
        default=None,
        help=(
            "Coverage solver to benchmark. Repeat for multiple solvers. "
            "Aliases: naive, greedy, lp, lp-relaxation, exact."
        ),
    )
    parser.add_argument(
        "--coverage-radius-km",
        nargs="+",
        type=float,
        default=list(DEFAULT_COVERAGE_RADII_KM),
    )
    parser.add_argument(
        "--facility-budget",
        nargs="+",
        type=int,
        default=list(DEFAULT_FACILITY_BUDGETS),
    )
    parser.add_argument("--holdout-hours", type=int, default=DEFAULT_HOLDOUT_HOURS)
    parser.add_argument("--test-months", type=int, default=DEFAULT_TEST_MONTHS)
    parser.add_argument("--max-hours", type=int, default=None)
    parser.add_argument("--background-rows", type=int, default=DEFAULT_BACKGROUND_ROWS)
    parser.add_argument(
        "--coalition-batch-size",
        type=int,
        default=DEFAULT_COALITION_BATCH_SIZE,
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--solver-seed", type=int, default=None)
    parser.add_argument("--benchmark-seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmup-hours", type=int, default=DEFAULT_WARMUP_HOURS)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--train-sample-rows", type=int, default=None)
    parser.add_argument("--xgb-n-estimators", type=int, default=100)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-verbosity", type=int, default=0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--mip-gap-abs", type=float, default=1e-9)
    parser.add_argument("--gurobi-threads", type=int, default=1)
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
        "--progress-every-coalitions",
        type=int,
        default=0,
        help=(
            "Progress logging interval inside each timed decision-SHAP task. "
            "Use 0 for clean benchmark timings."
        ),
    )
    parser.add_argument(
        "--no-gc-between-trials",
        dest="gc_between_trials",
        action="store_false",
        default=True,
        help="Do not call gc.collect() immediately before each timed trial.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    solvers = _resolve_solvers(args.solver)
    solver_seed = args.random_state if args.solver_seed is None else args.solver_seed
    base_config = _build_base_config(args, solver_seed=solver_seed)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("Preparing EMS/XGBoost benchmark inputs.", flush=True)
    prepared = _prepare_benchmark_data(base_config)
    settings = _build_benchmark_settings(
        args.coverage_radius_km,
        args.facility_budget,
        distance_matrix_path=args.distance_matrix_path,
        zip_codes=prepared.zip_codes,
    )
    config_by_setting_solver = {
        (setting.setting_id, solver): replace(
            base_config,
            coverage_radius_km=setting.coverage_radius_km,
            facility_budget=setting.facility_budget,
            coverage_solver=solver,
        )
        for setting in settings
        for solver in solvers
    }

    if args.warmup_hours > 0:
        print(
            f"Running {args.warmup_hours} warmup hour(s) per setting/solver.",
            flush=True,
        )
        _run_warmups(
            prepared=prepared,
            settings=settings,
            solvers=solvers,
            config_by_setting_solver=config_by_setting_solver,
            warmup_hours=args.warmup_hours,
        )

    print(
        "Timing "
        f"{len(settings)} setting(s) x {len(solvers)} solver(s) x "
        f"{len(prepared.explained_timestamps)} hour(s) x {args.repetitions} "
        "repetition(s).",
        flush=True,
    )
    raw_timing = _run_timing_trials(
        prepared=prepared,
        settings=settings,
        solvers=solvers,
        config_by_setting_solver=config_by_setting_solver,
        repetitions=args.repetitions,
        benchmark_seed=args.benchmark_seed,
        gc_between_trials=args.gc_between_trials,
    )

    setting_summary = _summarize_timing(
        raw_timing,
        group_columns=("setting_id", "coverage_solver", "coverage_solver_label"),
        bootstrap_draws=args.bootstrap_draws,
        seed=args.benchmark_seed,
    )
    solver_summary = _summarize_timing(
        raw_timing,
        group_columns=("coverage_solver", "coverage_solver_label"),
        bootstrap_draws=args.bootstrap_draws,
        seed=args.benchmark_seed + 1,
    )
    speedup = _build_speedup_frame(raw_timing)
    speedup_summary = (
        _summarize_speedups(speedup, bootstrap_draws=args.bootstrap_draws, seed=args.benchmark_seed + 2)
        if not speedup.empty
        else pd.DataFrame()
    )
    predictive_summary = _summarize_predictive_timing(prepared.predictive_timing)

    raw_timing.to_csv(outdir / "raw_timing.csv", index=False)
    setting_summary.to_csv(outdir / "setting_summary.csv", index=False)
    solver_summary.to_csv(outdir / "solver_summary.csv", index=False)
    speedup.to_csv(outdir / "speedup_vs_exact.csv", index=False)
    speedup_summary.to_csv(outdir / "speedup_summary.csv", index=False)
    prepared.predictive_timing.to_csv(outdir / "predictive_coalition_timing.csv", index=False)
    predictive_summary.to_csv(outdir / "predictive_coalition_summary.csv", index=False)
    _write_metadata(
        outdir / "benchmark_metadata.json",
        args=args,
        solvers=solvers,
        settings=settings,
        prepared=prepared,
        raw_timing=raw_timing,
    )
    _write_markdown_report(
        outdir / "README.md",
        solver_summary=solver_summary,
        setting_summary=setting_summary,
        speedup_summary=speedup_summary,
        predictive_summary=predictive_summary,
        args=args,
    )

    print(f"Wrote raw benchmark timings to {outdir / 'raw_timing.csv'}", flush=True)
    print(f"Wrote benchmark summaries to {outdir}", flush=True)


def _validate_args(args: argparse.Namespace) -> None:
    if args.holdout_hours <= 0:
        raise ValueError("--holdout-hours must be strictly positive.")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be strictly positive.")
    if args.warmup_hours < 0:
        raise ValueError("--warmup-hours must be non-negative.")
    if args.bootstrap_draws < 0:
        raise ValueError("--bootstrap-draws must be non-negative.")
    if not args.coverage_radius_km:
        raise ValueError("At least one --coverage-radius-km is required.")
    if not args.facility_budget:
        raise ValueError("At least one --facility-budget is required.")


def _resolve_solvers(raw_solvers: Sequence[str] | None) -> tuple[str, ...]:
    if raw_solvers is None:
        return DEFAULT_SOLVERS
    normalized = []
    for solver_name in raw_solvers:
        solver_key = str(solver_name).strip().lower().replace("-", "_")
        solver_key = _SOLVER_ALIASES.get(solver_key, solver_key)
        normalized.append(normalize_ems_coverage_solver(solver_key))
    return tuple(dict.fromkeys(normalized))


def _build_base_config(args: argparse.Namespace, *, solver_seed: int) -> EmsExactShapConfig:
    config = EmsExactShapConfig(
        x_path=args.x_path,
        y_path=args.y_path,
        metadata_path=args.metadata_path,
        zone_order_path=args.zone_order_path,
        distance_matrix_path=args.distance_matrix_path,
        outdir=args.outdir,
        holdout_hours=args.holdout_hours,
        test_months=args.test_months,
        max_hours=args.max_hours,
        background_rows=args.background_rows,
        coalition_batch_size=args.coalition_batch_size,
        progress_every_coalitions=args.progress_every_coalitions,
        random_state=args.random_state,
        n_jobs=1,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_reg_lambda=args.xgb_reg_lambda,
        xgb_verbosity=args.xgb_verbosity,
        train_sample_rows=args.train_sample_rows,
        coverage_radius_km=float(args.coverage_radius_km[0]),
        facility_budget=int(args.facility_budget[0]),
        solver_seed=solver_seed,
        mip_gap=args.mip_gap,
        mip_gap_abs=args.mip_gap_abs,
        gurobi_threads=args.gurobi_threads,
        objective_tolerance=args.objective_tolerance,
        coverage_solver=DEFAULT_SOLVERS[0],
        excluded_zip_codes=tuple(str(zip_code) for zip_code in args.exclude_zip_codes),
        save_coalition_values=False,
        compute_cvar_decision_shap=False,
    )
    _validate_config(config)
    return config


def _prepare_benchmark_data(config: EmsExactShapConfig) -> BenchmarkPreparedData:
    started_ns = time.perf_counter_ns()
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
    print(
        f"Training XGBRegressor on {len(train_x):,} EMS hours; "
        f"explaining {len(explain_x):,} matched holdout hours.",
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

    coalition_predictions: list[np.ndarray] = []
    true_demands: list[np.ndarray] = []
    predictive_rows: list[dict[str, Any]] = []
    explained_timestamps = tuple(str(timestamp) for timestamp in explain_x[EMS_TIMESTAMP_COLUMN])
    for hour_idx, timestamp_key in enumerate(explained_timestamps):
        predict_started_ns = time.perf_counter_ns()
        predictions = coalition_predictor.predict_all_coalitions(
            explain_x.loc[hour_idx, list(feature_columns)],
            progress_label=None,
            progress_every_coalitions=0,
        )
        predict_seconds = _elapsed_seconds(predict_started_ns)
        coalition_predictions.append(predictions)
        true_demand = explain_y.loc[hour_idx, list(target_columns)].to_numpy(
            dtype=float,
            copy=True,
        )
        true_demands.append(true_demand)
        predictive_rows.append(
            {
                "hour_index": hour_idx,
                "timestamp_hour": timestamp_key,
                "xgb_coalition_prediction_seconds": predict_seconds,
                "coalition_count": int(predictions.shape[0]),
                "target_count": int(predictions.shape[1]),
                "background_rows": int(config.background_rows),
                "xgb_prediction_rows": int(predictions.shape[0] * config.background_rows),
            }
        )

    preparation_metadata = {
        "candidate_train_rows": int(unsampled_train_rows),
        "train_rows": int(len(train_x)),
        "holdout_rows": int(len(time_split.holdout_x)),
        "train_start": str(time_split.train_start),
        "train_end": str(time_split.train_end),
        "holdout_start": str(time_split.holdout_start),
        "holdout_end": str(time_split.holdout_end),
        "explained_rows": list(explained_rows),
        "training_seconds": training_seconds,
        "preparation_seconds": _elapsed_seconds(started_ns),
        "coalition_count": int(coalition_predictor.coalition_count),
        "player_count": int(len(player_names)),
        "zip_count": int(len(zip_codes)),
    }
    return BenchmarkPreparedData(
        config=config,
        feature_columns=tuple(feature_columns),
        player_names=player_names,
        target_columns=target_columns,
        zip_codes=zip_codes,
        explained_timestamps=explained_timestamps,
        true_demands=tuple(true_demands),
        coalition_predictions=tuple(coalition_predictions),
        predictive_timing=pd.DataFrame(predictive_rows),
        prediction_metrics=prediction_metrics,
        preparation_metadata=preparation_metadata,
    )


def _build_benchmark_settings(
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
    *,
    distance_matrix_path: Path,
    zip_codes: Sequence[str],
) -> tuple[BenchmarkSetting, ...]:
    distance_matrix = _load_distance_matrix(distance_matrix_path)
    settings: list[BenchmarkSetting] = []
    for radius_km in coverage_radii_km:
        coverage_matrix = build_coverage_matrix(
            distance_matrix,
            zip_codes,
            coverage_radius_km=float(radius_km),
        )
        density = float(np.mean(coverage_matrix))
        for facility_budget in facility_budgets:
            setting_id = (
                f"radius_{_format_numeric_id(float(radius_km))}km_"
                f"budget_{int(facility_budget)}"
            )
            settings.append(
                BenchmarkSetting(
                    setting_id=setting_id,
                    coverage_radius_km=float(radius_km),
                    facility_budget=int(facility_budget),
                    coverage_matrix=coverage_matrix,
                    coverage_matrix_density=density,
                )
            )
    return tuple(settings)


def _run_warmups(
    *,
    prepared: BenchmarkPreparedData,
    settings: Sequence[BenchmarkSetting],
    solvers: Sequence[str],
    config_by_setting_solver: dict[tuple[str, str], EmsExactShapConfig],
    warmup_hours: int,
) -> None:
    warmup_count = min(int(warmup_hours), len(prepared.explained_timestamps))
    for setting in settings:
        for solver in solvers:
            config = config_by_setting_solver[(setting.setting_id, solver)]
            for hour_idx in range(warmup_count):
                _time_one_attribution(
                    prepared=prepared,
                    setting=setting,
                    solver=solver,
                    config=config,
                    repetition=-1,
                    order_index=-1,
                    hour_idx=hour_idx,
                    gc_between_trials=False,
                )


def _run_timing_trials(
    *,
    prepared: BenchmarkPreparedData,
    settings: Sequence[BenchmarkSetting],
    solvers: Sequence[str],
    config_by_setting_solver: dict[tuple[str, str], EmsExactShapConfig],
    repetitions: int,
    benchmark_seed: int,
    gc_between_trials: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(benchmark_seed)
    rows: list[dict[str, Any]] = []
    tasks = [
        (setting, solver, hour_idx)
        for setting in settings
        for solver in solvers
        for hour_idx in range(len(prepared.explained_timestamps))
    ]
    total_trials = len(tasks) * repetitions
    completed_trials = 0
    report_every = max(1, total_trials // 20)
    for repetition in range(repetitions):
        order = rng.permutation(len(tasks))
        for order_index, task_index in enumerate(order):
            setting, solver, hour_idx = tasks[int(task_index)]
            row = _time_one_attribution(
                prepared=prepared,
                setting=setting,
                solver=solver,
                config=config_by_setting_solver[(setting.setting_id, solver)],
                repetition=repetition,
                order_index=order_index,
                hour_idx=hour_idx,
                gc_between_trials=gc_between_trials,
            )
            rows.append(row)
            completed_trials += 1
            if completed_trials == total_trials or completed_trials % report_every == 0:
                print(
                    f"Completed {completed_trials:,}/{total_trials:,} timed trials.",
                    flush=True,
                )
    return pd.DataFrame(rows)


def _time_one_attribution(
    *,
    prepared: BenchmarkPreparedData,
    setting: BenchmarkSetting,
    solver: str,
    config: EmsExactShapConfig,
    repetition: int,
    order_index: int,
    hour_idx: int,
    gc_between_trials: bool,
) -> dict[str, Any]:
    if gc_between_trials:
        gc.collect()
    timestamp_key = prepared.explained_timestamps[hour_idx]
    solve_started_ns = time.perf_counter_ns()
    decision_values, baseline_solution, full_solution, _decision_solutions = (
        _solve_decision_values(
            coalition_demand_matrix=prepared.coalition_predictions[hour_idx],
            true_demand=prepared.true_demands[hour_idx],
            coverage_matrix=setting.coverage_matrix,
            zip_codes=prepared.zip_codes,
            config=config,
            coverage_solver=solver,
            progress_label=None,
        )
    )
    solve_seconds = _elapsed_seconds(solve_started_ns)
    shap_started_ns = time.perf_counter_ns()
    characteristic_values = decision_values - decision_values[0]
    decision_shap = compute_exact_shapley_values(
        characteristic_values,
        feature_count=len(prepared.player_names),
    )
    shap_seconds = _elapsed_seconds(shap_started_ns)
    attribution_seconds = solve_seconds + shap_seconds
    full_minus_baseline = float(characteristic_values[-1])
    shap_sum = float(np.sum(decision_shap))
    return {
        "repetition": int(repetition),
        "order_index": int(order_index),
        "setting_id": setting.setting_id,
        "coverage_radius_km": float(setting.coverage_radius_km),
        "facility_budget": int(setting.facility_budget),
        "coverage_matrix_density": float(setting.coverage_matrix_density),
        "coverage_solver": solver,
        "coverage_solver_label": _SOLVER_LABELS.get(solver, solver),
        "hour_index": int(hour_idx),
        "timestamp_hour": timestamp_key,
        "decision_solve_seconds": solve_seconds,
        "shap_transform_seconds": shap_seconds,
        "attribution_seconds": attribution_seconds,
        "coalitions_solved": int(len(decision_values)),
        "seconds_per_coalition": float(solve_seconds / max(1, len(decision_values))),
        "player_count": int(len(prepared.player_names)),
        "zip_count": int(len(prepared.zip_codes)),
        "baseline_value": float(decision_values[0]),
        "full_value": float(decision_values[-1]),
        "full_minus_baseline_value": full_minus_baseline,
        "shap_sum": shap_sum,
        "shap_additivity_abs_error": abs(shap_sum - full_minus_baseline),
        "baseline_selected_facility_indices": json.dumps(
            list(baseline_solution.selected_facility_indices)
        ),
        "full_selected_facility_indices": json.dumps(
            list(full_solution.selected_facility_indices)
        ),
    }


def _summarize_timing(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for group_values, group in frame.groupby(list(group_columns), sort=True, observed=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {
            column: value
            for column, value in zip(group_columns, group_values, strict=True)
        }
        row.update(_timing_stats(group["attribution_seconds"].to_numpy(dtype=float), rng, bootstrap_draws))
        row.update(
            {
                "mean_decision_solve_seconds": float(group["decision_solve_seconds"].mean()),
                "median_decision_solve_seconds": float(group["decision_solve_seconds"].median()),
                "mean_shap_transform_seconds": float(group["shap_transform_seconds"].mean()),
                "median_shap_transform_seconds": float(group["shap_transform_seconds"].median()),
                "mean_seconds_per_coalition": float(group["seconds_per_coalition"].mean()),
                "max_shap_additivity_abs_error": float(
                    group["shap_additivity_abs_error"].max()
                ),
            }
        )
        for metadata_column in (
            "coverage_radius_km",
            "facility_budget",
            "coverage_matrix_density",
            "coalitions_solved",
            "player_count",
            "zip_count",
        ):
            if metadata_column in group.columns and metadata_column not in row:
                unique_values = group[metadata_column].drop_duplicates()
                if len(unique_values) == 1:
                    row[metadata_column] = unique_values.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _timing_stats(
    values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_draws: int,
) -> dict[str, float | int]:
    cleaned = np.asarray(values, dtype=float)
    if cleaned.size == 0:
        raise ValueError("Cannot summarize an empty timing sample.")
    mean_ci_low, mean_ci_high = _bootstrap_mean_ci(cleaned, rng, bootstrap_draws)
    q05, q25, q75, q95 = np.quantile(cleaned, [0.05, 0.25, 0.75, 0.95])
    return {
        "n": int(cleaned.size),
        "mean_attribution_seconds": float(np.mean(cleaned)),
        "mean_attribution_seconds_ci95_low": mean_ci_low,
        "mean_attribution_seconds_ci95_high": mean_ci_high,
        "std_attribution_seconds": float(np.std(cleaned, ddof=1)) if cleaned.size > 1 else 0.0,
        "median_attribution_seconds": float(np.median(cleaned)),
        "q05_attribution_seconds": float(q05),
        "q25_attribution_seconds": float(q25),
        "q75_attribution_seconds": float(q75),
        "q95_attribution_seconds": float(q95),
        "min_attribution_seconds": float(np.min(cleaned)),
        "max_attribution_seconds": float(np.max(cleaned)),
    }


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float]:
    cleaned = np.asarray(values, dtype=float)
    if draws <= 0 or cleaned.size <= 1:
        mean_value = float(np.mean(cleaned))
        return mean_value, mean_value
    sample_indices = rng.integers(0, cleaned.size, size=(int(draws), cleaned.size))
    means = cleaned[sample_indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _build_speedup_frame(raw_timing: pd.DataFrame) -> pd.DataFrame:
    exact = raw_timing.loc[
        raw_timing["coverage_solver"].eq("gurobi"),
        ["repetition", "setting_id", "hour_index", "attribution_seconds"],
    ].rename(columns={"attribution_seconds": "exact_attribution_seconds"})
    if exact.empty:
        return pd.DataFrame()
    merged = raw_timing.merge(
        exact,
        on=["repetition", "setting_id", "hour_index"],
        how="inner",
    )
    merged["speedup_vs_exact"] = (
        merged["exact_attribution_seconds"] / merged["attribution_seconds"]
    )
    return merged[
        [
            "repetition",
            "setting_id",
            "coverage_radius_km",
            "facility_budget",
            "coverage_solver",
            "coverage_solver_label",
            "hour_index",
            "timestamp_hour",
            "attribution_seconds",
            "exact_attribution_seconds",
            "speedup_vs_exact",
        ]
    ].copy()


def _summarize_speedups(
    speedup: pd.DataFrame,
    *,
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    group_columns = ("setting_id", "coverage_solver", "coverage_solver_label")
    for group_values, group in speedup.groupby(list(group_columns), sort=True, observed=True):
        values = group["speedup_vs_exact"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(values, rng, bootstrap_draws)
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {
            column: value
            for column, value in zip(group_columns, group_values, strict=True)
        }
        row.update(
            {
                "n": int(len(values)),
                "mean_speedup_vs_exact": float(np.mean(values)),
                "mean_speedup_vs_exact_ci95_low": ci_low,
                "mean_speedup_vs_exact_ci95_high": ci_high,
                "median_speedup_vs_exact": float(np.median(values)),
                "q25_speedup_vs_exact": float(np.quantile(values, 0.25)),
                "q75_speedup_vs_exact": float(np.quantile(values, 0.75)),
            }
        )
        for metadata_column in ("coverage_radius_km", "facility_budget"):
            unique_values = group[metadata_column].drop_duplicates()
            if len(unique_values) == 1:
                row[metadata_column] = unique_values.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_predictive_timing(predictive_timing: pd.DataFrame) -> pd.DataFrame:
    values = predictive_timing["xgb_coalition_prediction_seconds"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "n": int(len(values)),
                "mean_xgb_coalition_prediction_seconds": float(np.mean(values)),
                "median_xgb_coalition_prediction_seconds": float(np.median(values)),
                "q25_xgb_coalition_prediction_seconds": float(np.quantile(values, 0.25)),
                "q75_xgb_coalition_prediction_seconds": float(np.quantile(values, 0.75)),
                "min_xgb_coalition_prediction_seconds": float(np.min(values)),
                "max_xgb_coalition_prediction_seconds": float(np.max(values)),
                "mean_xgb_prediction_rows": float(
                    predictive_timing["xgb_prediction_rows"].mean()
                ),
            }
        ]
    )


def _write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    solvers: Sequence[str],
    settings: Sequence[BenchmarkSetting],
    prepared: BenchmarkPreparedData,
    raw_timing: pd.DataFrame,
) -> None:
    metadata = {
        "benchmark_design": {
            "timed_region": (
                "Solver-specific EMS decision-SHAP attribution: solve maximum-coverage "
                "decision values for every exact grouped-SHAP coalition, then run the "
                "exact Shapley transform."
            ),
            "excluded_from_solver_comparison": (
                "XGBoost training, holdout prediction, empirical-background coalition "
                "prediction, plotting, CSV writing, and CVaR decision SHAP."
            ),
            "matched_design": (
                "All solvers use the same fitted XGBoost model, same background rows, "
                "same sampled holdout hours, same coalition predictions, same true "
                "EMS demands, and same coverage matrix for a setting."
            ),
            "randomization": (
                "Within each repetition, setting/solver/hour timing tasks are run in "
                "a seeded random order."
            ),
            "implementation_scope": (
                "Measures the repository implementation as used by the exhaustive "
                "comparison; no optimizer-model reuse is introduced by this benchmark."
            ),
        },
        "arguments": _jsonable_vars(args),
        "solvers": list(solvers),
        "solver_labels": {solver: _SOLVER_LABELS.get(solver, solver) for solver in solvers},
        "settings": [
            {
                "setting_id": setting.setting_id,
                "coverage_radius_km": setting.coverage_radius_km,
                "facility_budget": setting.facility_budget,
                "coverage_matrix_density": setting.coverage_matrix_density,
            }
            for setting in settings
        ],
        "prepared_data": prepared.preparation_metadata,
        "prediction_metrics": prepared.prediction_metrics,
        "model": {
            "name": "XGBRegressor",
            "xgb_params": {
                "n_estimators": prepared.config.xgb_n_estimators,
                "max_depth": prepared.config.xgb_max_depth,
                "learning_rate": prepared.config.xgb_learning_rate,
                "subsample": prepared.config.xgb_subsample,
                "colsample_bytree": prepared.config.xgb_colsample_bytree,
                "reg_lambda": prepared.config.xgb_reg_lambda,
                "tree_method": "hist",
                "n_jobs": 1,
                "random_state": prepared.config.random_state,
            },
        },
        "timing": {
            "timed_trial_count": int(len(raw_timing)),
            "timer": "time.perf_counter_ns",
            "gc_between_trials": bool(args.gc_between_trials),
            "repetitions": int(args.repetitions),
            "warmup_hours_per_setting_solver": int(args.warmup_hours),
            "bootstrap_draws": int(args.bootstrap_draws),
        },
        "software": _software_metadata(),
        "system": _system_metadata(),
        "git": _git_metadata(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _write_markdown_report(
    path: Path,
    *,
    solver_summary: pd.DataFrame,
    setting_summary: pd.DataFrame,
    speedup_summary: pd.DataFrame,
    predictive_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# EMS Decision-SHAP Timing Benchmark",
        "",
        "This benchmark times solver-specific EMS decision-SHAP attribution for the "
        "four maximum-coverage methods used in the exhaustive comparison. XGBoost "
        "coalition prediction is timed separately because it is shared across solvers.",
        "",
        f"- Holdout hours: {args.holdout_hours}",
        f"- Repetitions: {args.repetitions}",
        f"- Warmup hours per setting/solver: {args.warmup_hours}",
        f"- Bootstrap draws: {args.bootstrap_draws}",
        "",
        "## Solver Summary",
        "",
        _markdown_table(
            solver_summary,
            [
                "coverage_solver_label",
                "n",
                "mean_attribution_seconds",
                "mean_attribution_seconds_ci95_low",
                "mean_attribution_seconds_ci95_high",
                "median_attribution_seconds",
                "mean_seconds_per_coalition",
            ],
        ),
        "",
    ]
    if not speedup_summary.empty:
        lines.extend(
            [
                "## Speedup Versus Exact",
                "",
                _markdown_table(
                    speedup_summary,
                    [
                        "setting_id",
                        "coverage_solver_label",
                        "n",
                        "mean_speedup_vs_exact",
                        "mean_speedup_vs_exact_ci95_low",
                        "mean_speedup_vs_exact_ci95_high",
                        "median_speedup_vs_exact",
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## XGBoost Coalition Prediction",
            "",
            _markdown_table(
                predictive_summary,
                [
                    "n",
                    "mean_xgb_coalition_prediction_seconds",
                    "median_xgb_coalition_prediction_seconds",
                    "mean_xgb_prediction_rows",
                ],
            ),
            "",
            "## Setting Summary",
            "",
            _markdown_table(
                setting_summary,
                [
                    "setting_id",
                    "coverage_solver_label",
                    "n",
                    "mean_attribution_seconds",
                    "median_attribution_seconds",
                    "mean_decision_solve_seconds",
                    "mean_shap_transform_seconds",
                ],
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    selected_columns = [column for column in columns if column in frame.columns]
    rows = [selected_columns]
    rows.extend(
        [
            [_format_markdown_value(row[column]) for column in selected_columns]
            for _, row in frame.loc[:, selected_columns].iterrows()
        ]
    )
    header = "| " + " | ".join(str(value) for value in rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in selected_columns) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows[1:]]
    return "\n".join([header, separator, *body])


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.6f}"
    return str(value)


def _jsonable_vars(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = value
    return result


def _software_metadata() -> dict[str, Any]:
    packages = {}
    for package_name in ("numpy", "pandas", "scikit-learn", "xgboost", "gurobipy"):
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": sys.version,
        "packages": packages,
    }


def _system_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
    }


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _run_git_command(("rev-parse", "HEAD")),
        "branch": _run_git_command(("rev-parse", "--abbrev-ref", "HEAD")),
        "status_short": _run_git_command(("status", "--short")),
    }


def _run_git_command(args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _format_numeric_id(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _elapsed_seconds(started_ns: int) -> float:
    return float((time.perf_counter_ns() - started_ns) / 1_000_000_000.0)


if __name__ == "__main__":
    main()
