from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dva.analysis.ems_exact_shap import (
    DEFAULT_COALITION_BATCH_SIZE,
    DEFAULT_DISTANCE_MATRIX_PATH,
    DEFAULT_EXCLUDED_ZIP_CODES,
    DEFAULT_FACILITY_BUDGET,
    DEFAULT_METADATA_PATH,
    DEFAULT_OBJECTIVE_TOLERANCE,
    DEFAULT_TEST_MONTHS,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_ZONE_ORDER_PATH,
    EMS_TIMESTAMP_COLUMN,
    EmsExactShapConfig,
    GroupedBackgroundCoalitionPredictor,
    _build_prediction_metrics,
    _build_solver_params,
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
    compute_kernel_shapley_values,
    compute_permutation_shapley_values,
    normalize_ems_coverage_solver,
)
from dva.analysis.evaluation_metrics import (
    build_attribution_ranking,
    compute_rank_kendall_tau_from_rankings,
)


DEFAULT_OUTPUT_DIR = Path("results/ems_decision_shap_approximation_experiment")
DEFAULT_HOLDOUT_HOURS = 100
DEFAULT_BACKGROUND_ROWS = 100
DEFAULT_COVERAGE_RADIUS_KM = 1.0
DEFAULT_SAMPLE_BUDGETS = (16, 32, 64, 128, 192)
DEFAULT_SEED_COUNT = 50
DEFAULT_SEED_START = 0
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_512
DEFAULT_METHODS = ("permutation", "kernel")
METHOD_IDS = {"permutation": 1, "kernel": 2}


@dataclass(frozen=True, slots=True)
class PreparedExperimentData:
    config: EmsExactShapConfig
    player_names: tuple[str, ...]
    target_columns: tuple[str, ...]
    zip_codes: tuple[str, ...]
    explained_timestamps: tuple[str, ...]
    true_demands: tuple[np.ndarray, ...]
    coalition_predictions: tuple[np.ndarray, ...]
    prediction_metrics: dict[str, Any]
    preparation_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExactReference:
    decision_games: tuple[np.ndarray, ...]
    exact_shap: np.ndarray
    timing: pd.DataFrame
    runtime_seconds: float
    oracle_calls: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the EMS deterministic decision-SHAP approximation experiment. "
            "Exact decision SHAP is computed once, then kernel/permutation "
            "approximations are compared across sampling budgets and seeds."
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
        "--coverage-radius-km",
        "--tau",
        dest="coverage_radius_km",
        type=float,
        default=DEFAULT_COVERAGE_RADIUS_KM,
        help="EMS coverage radius. Alias --tau is provided for the paper notation.",
    )
    parser.add_argument("--facility-budget", type=int, default=DEFAULT_FACILITY_BUDGET)
    parser.add_argument(
        "--solver",
        default="exact",
        help="Deterministic EMS coverage solver for the decision game.",
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
    parser.add_argument("--solver-threads", "--gurobi-threads", dest="gurobi_threads", type=int, default=1)
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
        "--method",
        action="append",
        choices=DEFAULT_METHODS,
        default=None,
        help="Approximation method to run. Repeat to select both explicitly.",
    )
    parser.add_argument(
        "--sample-budget",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAMPLE_BUDGETS),
        help="Approximation sampling budgets M.",
    )
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument(
        "--approximation-seed",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Explicit approximation seeds. When omitted, seeds are generated from "
            "--seed-start and --seed-count."
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--expected-player-count",
        type=int,
        default=8,
        help="Fail fast if the EMS feature grouping does not match the intended p.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    methods = tuple(args.method or DEFAULT_METHODS)
    sample_budgets = _normalize_positive_ints(args.sample_budget, "sample_budget")
    approximation_seeds = _resolve_approximation_seeds(args)
    solver_seed = args.random_state if args.solver_seed is None else args.solver_seed
    coverage_solver = normalize_ems_coverage_solver(args.solver)
    config = _build_config(args, solver_seed=solver_seed, coverage_solver=coverage_solver)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_experiment_data(config)
    if len(prepared.player_names) != args.expected_player_count:
        raise ValueError(
            f"Expected p={args.expected_player_count} EMS feature groups, "
            f"found p={len(prepared.player_names)}."
        )

    distance_matrix = _load_distance_matrix(config.distance_matrix_path)
    coverage_matrix = build_coverage_matrix(
        distance_matrix,
        prepared.zip_codes,
        coverage_radius_km=config.coverage_radius_km,
    )

    print(
        "Computing exact deterministic decision SHAP once "
        f"for {len(prepared.explained_timestamps)} EMS hours.",
        flush=True,
    )
    exact_reference = _compute_exact_reference(
        prepared=prepared,
        coverage_matrix=coverage_matrix,
        coverage_solver=coverage_solver,
    )

    exact_denominator = float(np.abs(exact_reference.exact_shap).sum())
    exact_hourly_shap = _build_exact_hourly_shap_frame(prepared, exact_reference)
    exact_metrics = _build_exact_metrics_row(
        exact_reference,
        prepared=prepared,
        exact_denominator=exact_denominator,
    )

    print(
        f"Running {len(methods)} method(s) x {len(sample_budgets)} budget(s) x "
        f"{len(approximation_seeds)} seed(s).",
        flush=True,
    )
    approximation_metrics = _run_approximation_grid(
        methods=methods,
        sample_budgets=sample_budgets,
        approximation_seeds=approximation_seeds,
        prepared=prepared,
        exact_reference=exact_reference,
        exact_denominator=exact_denominator,
        exact_seconds_per_oracle=_exact_seconds_per_oracle(exact_reference),
    )
    raw_metrics = pd.concat(
        [pd.DataFrame([exact_metrics]), approximation_metrics],
        ignore_index=True,
    )
    summary_metrics = _summarize_metrics(
        raw_metrics,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )

    exact_reference.timing.to_csv(outdir / "exact_reference_timing.csv", index=False)
    exact_hourly_shap.to_csv(outdir / "exact_hourly_shap.csv", index=False)
    approximation_metrics.to_csv(outdir / "raw_approximation_metrics.csv", index=False)
    raw_metrics.to_csv(outdir / "raw_metrics.csv", index=False)
    summary_metrics.to_csv(outdir / "summary_metrics.csv", index=False)
    _write_metadata(
        outdir / "run_metadata.json",
        args=args,
        methods=methods,
        sample_budgets=sample_budgets,
        approximation_seeds=approximation_seeds,
        prepared=prepared,
        exact_reference=exact_reference,
        coverage_matrix=coverage_matrix,
        exact_denominator=exact_denominator,
    )
    _write_readme(
        outdir / "README.md",
        args=args,
        raw_metrics=raw_metrics,
        summary_metrics=summary_metrics,
        exact_denominator=exact_denominator,
    )

    print(f"Wrote EMS approximation experiment outputs to {outdir}", flush=True)


def _validate_args(args: argparse.Namespace) -> None:
    if args.holdout_hours <= 0:
        raise ValueError("--holdout-hours must be strictly positive.")
    if args.test_months <= 0:
        raise ValueError("--test-months must be strictly positive.")
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("--max-hours must be strictly positive when provided.")
    if args.background_rows <= 0:
        raise ValueError("--background-rows must be strictly positive.")
    if args.seed_count <= 0 and args.approximation_seed is None:
        raise ValueError("--seed-count must be positive when explicit seeds are omitted.")
    if args.bootstrap_draws < 0:
        raise ValueError("--bootstrap-draws must be non-negative.")
    if args.expected_player_count <= 0:
        raise ValueError("--expected-player-count must be strictly positive.")
    _normalize_positive_ints(args.sample_budget, "sample_budget")


def _build_config(
    args: argparse.Namespace,
    *,
    solver_seed: int,
    coverage_solver: str,
) -> EmsExactShapConfig:
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
        progress_every_coalitions=0,
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
        coverage_radius_km=float(args.coverage_radius_km),
        facility_budget=int(args.facility_budget),
        solver_seed=solver_seed,
        mip_gap=args.mip_gap,
        mip_gap_abs=args.mip_gap_abs,
        gurobi_threads=args.gurobi_threads,
        optimization_solver=args.optimization_solver,
        objective_tolerance=args.objective_tolerance,
        coverage_solver=coverage_solver,
        excluded_zip_codes=tuple(str(zip_code) for zip_code in args.exclude_zip_codes),
        save_coalition_values=False,
        compute_cvar_decision_shap=False,
    )
    _validate_config(config)
    return config


def _prepare_experiment_data(config: EmsExactShapConfig) -> PreparedExperimentData:
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
        f"explaining {len(explain_x):,} holdout hours.",
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
    explained_timestamps = tuple(str(timestamp) for timestamp in explain_x[EMS_TIMESTAMP_COLUMN])
    for hour_idx, timestamp_key in enumerate(explained_timestamps):
        predictions = coalition_predictor.predict_all_coalitions(
            explain_x.loc[hour_idx, list(feature_columns)],
            progress_label=None,
            progress_every_coalitions=0,
        )
        coalition_predictions.append(predictions)
        true_demands.append(
            explain_y.loc[hour_idx, list(target_columns)].to_numpy(
                dtype=float,
                copy=True,
            )
        )
        if (hour_idx + 1) % max(1, len(explained_timestamps) // 10) == 0:
            print(
                f"Prepared coalition predictions for "
                f"{hour_idx + 1:,}/{len(explained_timestamps):,} hours.",
                flush=True,
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
    return PreparedExperimentData(
        config=config,
        player_names=player_names,
        target_columns=target_columns,
        zip_codes=zip_codes,
        explained_timestamps=explained_timestamps,
        true_demands=tuple(true_demands),
        coalition_predictions=tuple(coalition_predictions),
        prediction_metrics=prediction_metrics,
        preparation_metadata=preparation_metadata,
    )


def _compute_exact_reference(
    *,
    prepared: PreparedExperimentData,
    coverage_matrix: np.ndarray,
    coverage_solver: str,
) -> ExactReference:
    exact_started_ns = time.perf_counter_ns()
    decision_games: list[np.ndarray] = []
    exact_shap_rows: list[np.ndarray] = []
    timing_rows: list[dict[str, Any]] = []
    config = prepared.config
    coalition_count = 1 << len(prepared.player_names)
    for hour_idx, timestamp_key in enumerate(prepared.explained_timestamps):
        solve_started_ns = time.perf_counter_ns()
        decision_values, _baseline_solution, _full_solution, _decision_solutions = (
            _solve_decision_values(
                coalition_demand_matrix=prepared.coalition_predictions[hour_idx],
                true_demand=prepared.true_demands[hour_idx],
                coverage_matrix=coverage_matrix,
                zip_codes=prepared.zip_codes,
                config=config,
                coverage_solver=coverage_solver,
                progress_label=None,
            )
        )
        solve_seconds = _elapsed_seconds(solve_started_ns)
        characteristic_values = decision_values - decision_values[0]
        shap_started_ns = time.perf_counter_ns()
        exact_shap = compute_exact_shapley_values(
            characteristic_values,
            feature_count=len(prepared.player_names),
        )
        shap_seconds = _elapsed_seconds(shap_started_ns)
        decision_games.append(characteristic_values)
        exact_shap_rows.append(exact_shap)
        timing_rows.append(
            {
                "hour_index": hour_idx,
                "timestamp_hour": timestamp_key,
                "exact_decision_solve_seconds": solve_seconds,
                "exact_shap_transform_seconds": shap_seconds,
                "exact_total_seconds": solve_seconds + shap_seconds,
                "decision_value_gain": float(characteristic_values[-1]),
                "oracle_calls": coalition_count,
            }
        )
        if (hour_idx + 1) % max(1, len(prepared.explained_timestamps) // 10) == 0:
            print(
                f"Solved exact decision games for "
                f"{hour_idx + 1:,}/{len(prepared.explained_timestamps):,} hours.",
                flush=True,
            )

    timing = pd.DataFrame(timing_rows)
    return ExactReference(
        decision_games=tuple(decision_games),
        exact_shap=np.vstack(exact_shap_rows),
        timing=timing,
        runtime_seconds=_elapsed_seconds(exact_started_ns),
        oracle_calls=int(coalition_count * len(prepared.explained_timestamps)),
    )


def _run_approximation_grid(
    *,
    methods: Sequence[str],
    sample_budgets: Sequence[int],
    approximation_seeds: Sequence[int],
    prepared: PreparedExperimentData,
    exact_reference: ExactReference,
    exact_denominator: float,
    exact_seconds_per_oracle: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_runs = len(methods) * len(sample_budgets) * len(approximation_seeds)
    completed_runs = 0
    report_every = max(1, total_runs // 20)
    for method in methods:
        for sample_budget in sample_budgets:
            for seed in approximation_seeds:
                approx_started_ns = time.perf_counter_ns()
                approximations = np.vstack(
                    [
                        _compute_one_approximation(
                            method,
                            decision_game,
                            feature_count=len(prepared.player_names),
                            sample_budget=sample_budget,
                            seed=seed,
                            hour_idx=hour_idx + 1,
                        )
                        for hour_idx, decision_game in enumerate(
                            exact_reference.decision_games
                        )
                    ]
                )
                transform_runtime_seconds = _elapsed_seconds(approx_started_ns)
                metric_values = _compute_approximation_metrics(
                    approximations,
                    exact_reference.exact_shap,
                    prepared.player_names,
                    exact_denominator=exact_denominator,
                )
                oracle_call_counts = _approximation_oracle_calls(
                    method,
                    feature_count=len(prepared.player_names),
                    sample_budget=sample_budget,
                    seed=seed,
                    hour_count=len(prepared.explained_timestamps),
                )
                estimated_cached_oracle_runtime_seconds = (
                    oracle_call_counts["unique"] * exact_seconds_per_oracle
                )
                estimated_noncached_oracle_runtime_seconds = (
                    oracle_call_counts["noncached"] * exact_seconds_per_oracle
                )
                estimated_standalone_cached_runtime_seconds = (
                    estimated_cached_oracle_runtime_seconds + transform_runtime_seconds
                )
                estimated_standalone_noncached_runtime_seconds = (
                    estimated_noncached_oracle_runtime_seconds
                    + transform_runtime_seconds
                )
                rows.append(
                    {
                        "method": method,
                        "sample_budget": int(sample_budget),
                        "seed": int(seed),
                        "runtime_seconds": estimated_standalone_cached_runtime_seconds,
                        "runtime_seconds_kind": (
                            "estimated_standalone_cached_from_exact_oracle_rate"
                        ),
                        "actual_runner_runtime_seconds": transform_runtime_seconds,
                        "transform_runtime_seconds": transform_runtime_seconds,
                        "estimated_oracle_runtime_seconds": (
                            estimated_cached_oracle_runtime_seconds
                        ),
                        "estimated_standalone_runtime_seconds": (
                            estimated_standalone_cached_runtime_seconds
                        ),
                        "estimated_cached_oracle_runtime_seconds": (
                            estimated_cached_oracle_runtime_seconds
                        ),
                        "estimated_noncached_oracle_runtime_seconds": (
                            estimated_noncached_oracle_runtime_seconds
                        ),
                        "estimated_standalone_cached_runtime_seconds": (
                            estimated_standalone_cached_runtime_seconds
                        ),
                        "estimated_standalone_noncached_runtime_seconds": (
                            estimated_standalone_noncached_runtime_seconds
                        ),
                        "exact_seconds_per_oracle_reference": exact_seconds_per_oracle,
                        "oracle_calls": oracle_call_counts["unique"],
                        "standalone_cached_oracle_calls": oracle_call_counts["unique"],
                        "standalone_unique_oracle_calls": oracle_call_counts["unique"],
                        "standalone_noncached_oracle_calls": oracle_call_counts[
                            "noncached"
                        ],
                        "actual_runner_oracle_calls": 0,
                        "hour_count": len(prepared.explained_timestamps),
                        "player_count": len(prepared.player_names),
                        **metric_values,
                    }
                )
                completed_runs += 1
                if completed_runs == total_runs or completed_runs % report_every == 0:
                    print(
                        f"Completed {completed_runs:,}/{total_runs:,} "
                        "approximation runs.",
                        flush=True,
                    )
    return pd.DataFrame(rows)


def _exact_seconds_per_oracle(exact_reference: ExactReference) -> float:
    exact_solve_seconds = float(
        exact_reference.timing["exact_decision_solve_seconds"].sum()
    )
    if exact_reference.oracle_calls <= 0:
        return 0.0
    return exact_solve_seconds / float(exact_reference.oracle_calls)


def _compute_one_approximation(
    method: str,
    decision_game: np.ndarray,
    *,
    feature_count: int,
    sample_budget: int,
    seed: int,
    hour_idx: int,
) -> np.ndarray:
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(hour_idx), METHOD_IDS[method], int(sample_budget)]
    )
    if method == "permutation":
        return compute_permutation_shapley_values(
            decision_game,
            feature_count,
            sample_count=sample_budget,
            random_state=seed_sequence,
        )
    if method == "kernel":
        return compute_kernel_shapley_values(
            decision_game,
            feature_count,
            sample_count=sample_budget,
            random_state=seed_sequence,
        )
    raise ValueError(f"Unsupported approximation method: {method}")


def _compute_approximation_metrics(
    approximation_shap: np.ndarray,
    exact_shap: np.ndarray,
    player_names: Sequence[str],
    *,
    exact_denominator: float,
) -> dict[str, float | int | None | str]:
    if approximation_shap.shape != exact_shap.shape:
        raise ValueError(
            "approximation_shap and exact_shap must share the same shape. "
            f"Got {approximation_shap.shape} and {exact_shap.shape}."
        )
    numerator = float(np.abs(approximation_shap - exact_shap).sum())
    nmae = None if exact_denominator <= 0.0 else float(numerator / exact_denominator)
    exact_importance = np.abs(exact_shap).mean(axis=0)
    approximation_importance = np.abs(approximation_shap).mean(axis=0)
    exact_ranking = build_attribution_ranking(exact_importance, player_names)
    approximation_ranking = build_attribution_ranking(
        approximation_importance,
        player_names,
    )
    return {
        "nmae": nmae,
        "absolute_error_sum": numerator,
        "exact_abs_shap_sum": float(exact_denominator),
        "global_top1_match": int(exact_ranking[0] == approximation_ranking[0]),
        "global_kendall_tau_b": compute_rank_kendall_tau_from_rankings(
            exact_ranking,
            approximation_ranking,
        ),
        "exact_global_top1_feature": exact_ranking[0],
        "approx_global_top1_feature": approximation_ranking[0],
    }


def _build_exact_metrics_row(
    exact_reference: ExactReference,
    *,
    prepared: PreparedExperimentData,
    exact_denominator: float,
) -> dict[str, Any]:
    exact_decision_solve_seconds = float(
        exact_reference.timing["exact_decision_solve_seconds"].sum()
    )
    exact_transform_seconds = float(
        exact_reference.timing["exact_shap_transform_seconds"].sum()
    )
    exact_seconds_per_oracle = _exact_seconds_per_oracle(exact_reference)
    exact_ranking = build_attribution_ranking(
        np.abs(exact_reference.exact_shap).mean(axis=0),
        prepared.player_names,
    )
    return {
        "method": "exact",
        "sample_budget": None,
        "seed": None,
        "runtime_seconds": float(exact_reference.runtime_seconds),
        "runtime_seconds_kind": "measured_standalone",
        "actual_runner_runtime_seconds": float(exact_reference.runtime_seconds),
        "transform_runtime_seconds": exact_transform_seconds,
        "estimated_oracle_runtime_seconds": exact_decision_solve_seconds,
        "estimated_standalone_runtime_seconds": float(exact_reference.runtime_seconds),
        "estimated_cached_oracle_runtime_seconds": exact_decision_solve_seconds,
        "estimated_noncached_oracle_runtime_seconds": exact_decision_solve_seconds,
        "estimated_standalone_cached_runtime_seconds": float(
            exact_reference.runtime_seconds
        ),
        "estimated_standalone_noncached_runtime_seconds": float(
            exact_reference.runtime_seconds
        ),
        "exact_seconds_per_oracle_reference": exact_seconds_per_oracle,
        "oracle_calls": int(exact_reference.oracle_calls),
        "standalone_cached_oracle_calls": int(exact_reference.oracle_calls),
        "standalone_unique_oracle_calls": int(exact_reference.oracle_calls),
        "standalone_noncached_oracle_calls": int(exact_reference.oracle_calls),
        "actual_runner_oracle_calls": int(exact_reference.oracle_calls),
        "hour_count": len(prepared.explained_timestamps),
        "player_count": len(prepared.player_names),
        "nmae": 0.0,
        "absolute_error_sum": 0.0,
        "exact_abs_shap_sum": float(exact_denominator),
        "global_top1_match": 1,
        "global_kendall_tau_b": 1.0,
        "exact_global_top1_feature": exact_ranking[0],
        "approx_global_top1_feature": exact_ranking[0],
    }


def _build_exact_hourly_shap_frame(
    prepared: PreparedExperimentData,
    exact_reference: ExactReference,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hour_idx, timestamp_key in enumerate(prepared.explained_timestamps):
        row: dict[str, Any] = {
            "hour_index": hour_idx,
            "timestamp_hour": timestamp_key,
        }
        for player_name, shap_value in zip(
            prepared.player_names,
            exact_reference.exact_shap[hour_idx],
            strict=True,
        ):
            row[f"decision_shap_{player_name}"] = float(shap_value)
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_metrics(
    raw_metrics: pd.DataFrame,
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    metric_columns = (
        "nmae",
        "runtime_seconds",
        "actual_runner_runtime_seconds",
        "transform_runtime_seconds",
        "estimated_oracle_runtime_seconds",
        "estimated_standalone_runtime_seconds",
        "estimated_cached_oracle_runtime_seconds",
        "estimated_noncached_oracle_runtime_seconds",
        "estimated_standalone_cached_runtime_seconds",
        "estimated_standalone_noncached_runtime_seconds",
        "global_top1_match",
        "global_kendall_tau_b",
        "oracle_calls",
        "standalone_cached_oracle_calls",
        "standalone_unique_oracle_calls",
        "standalone_noncached_oracle_calls",
        "actual_runner_oracle_calls",
    )
    rows: list[dict[str, Any]] = []
    group_frame = raw_metrics.copy()
    group_frame["sample_budget"] = group_frame["sample_budget"].astype("Int64")
    for (method, sample_budget), group in group_frame.groupby(
        ["method", "sample_budget"],
        dropna=False,
        sort=True,
    ):
        row: dict[str, Any] = {
            "method": method,
            "sample_budget": None if pd.isna(sample_budget) else int(sample_budget),
            "n_runs": int(len(group)),
        }
        for metric_column in metric_columns:
            stats = _bootstrap_mean_summary(
                group[metric_column].to_numpy(dtype=float),
                rng=rng,
                draws=bootstrap_draws,
            )
            row[f"{metric_column}_mean"] = stats["mean"]
            row[f"{metric_column}_ci95_low"] = stats["ci95_low"]
            row[f"{metric_column}_ci95_high"] = stats["ci95_high"]
            row[f"{metric_column}_std"] = stats["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean_summary(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, float | None]:
    cleaned = np.asarray(values, dtype=float)
    cleaned = cleaned[np.isfinite(cleaned)]
    if cleaned.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "std": None}
    mean_value = float(np.mean(cleaned))
    std_value = float(np.std(cleaned, ddof=1)) if cleaned.size > 1 else 0.0
    if draws <= 0 or cleaned.size <= 1:
        return {
            "mean": mean_value,
            "ci95_low": mean_value,
            "ci95_high": mean_value,
            "std": std_value,
        }
    sample_indices = rng.integers(0, cleaned.size, size=(int(draws), cleaned.size))
    bootstrap_means = cleaned[sample_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "mean": mean_value,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "std": std_value,
    }


def _approximation_oracle_calls(
    method: str,
    *,
    feature_count: int,
    sample_budget: int,
    seed: int,
    hour_count: int,
) -> dict[str, int]:
    if method == "permutation":
        unique_calls = sum(
            _permutation_unique_masks(
                feature_count=feature_count,
                sample_budget=sample_budget,
                seed=seed,
                hour_idx=hour_idx + 1,
            )
            for hour_idx in range(hour_count)
        )
        noncached_calls = int(hour_count * sample_budget * (feature_count + 1))
        return {"unique": int(unique_calls), "noncached": noncached_calls}
    if method == "kernel":
        unique_calls = sum(
            _kernel_unique_masks(
                feature_count=feature_count,
                sample_budget=sample_budget,
                seed=seed,
                hour_idx=hour_idx + 1,
            )
            for hour_idx in range(hour_count)
        )
        noncached_calls = int(hour_count * (sample_budget + 2))
        return {"unique": int(unique_calls), "noncached": noncached_calls}
    raise ValueError(f"Unsupported approximation method: {method}")


def _permutation_unique_masks(
    *,
    feature_count: int,
    sample_budget: int,
    seed: int,
    hour_idx: int,
) -> int:
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(hour_idx), METHOD_IDS["permutation"], int(sample_budget)]
    )
    rng = np.random.default_rng(seed_sequence)
    masks = {0}
    for _ in range(sample_budget):
        coalition_mask = 0
        masks.add(coalition_mask)
        for feature_idx in rng.permutation(feature_count):
            coalition_mask |= 1 << int(feature_idx)
            masks.add(coalition_mask)
    return len(masks)


def _kernel_unique_masks(
    *,
    feature_count: int,
    sample_budget: int,
    seed: int,
    hour_idx: int,
) -> int:
    if feature_count <= 1:
        return 2
    full_mask = (1 << feature_count) - 1
    interior_mask_count = full_mask - 1
    if sample_budget <= interior_mask_count:
        return sample_budget + 2
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(hour_idx), METHOD_IDS["kernel"], int(sample_budget)]
    )
    rng = np.random.default_rng(seed_sequence)
    interior_masks = np.arange(1, full_mask, dtype=int)
    sampled_masks = rng.choice(interior_masks, size=sample_budget, replace=True)
    return len({0, full_mask, *(int(mask) for mask in sampled_masks)})


def _normalize_positive_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must contain positive integers.")
        integer_value = int(value)
        if integer_value in seen:
            continue
        seen.add(integer_value)
        normalized.append(integer_value)
    if not normalized:
        raise ValueError(f"{name} must contain at least one value.")
    return tuple(normalized)


def _resolve_approximation_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.approximation_seed is not None:
        return _normalize_ints(args.approximation_seed, "approximation_seed")
    return tuple(range(int(args.seed_start), int(args.seed_start) + int(args.seed_count)))


def _normalize_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain integers.")
        integer_value = int(value)
        if integer_value in seen:
            continue
        seen.add(integer_value)
        normalized.append(integer_value)
    if not normalized:
        raise ValueError(f"{name} must contain at least one value.")
    return tuple(normalized)


def _elapsed_seconds(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000_000.0


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in values.items()
    }


def _write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    methods: Sequence[str],
    sample_budgets: Sequence[int],
    approximation_seeds: Sequence[int],
    prepared: PreparedExperimentData,
    exact_reference: ExactReference,
    coverage_matrix: np.ndarray,
    exact_denominator: float,
) -> None:
    metadata = {
        "experiment": {
            "name": "ems_decision_shap_approximation_experiment",
            "description": (
                "Exact deterministic decision SHAP is computed once. Kernel and "
                "permutation SHAP approximations reuse the exact decision game values "
                "for fair metric computation; oracle call counts report standalone "
                "coalition-value requirements."
            ),
            "methods": list(methods),
            "sample_budgets": [int(value) for value in sample_budgets],
            "approximation_seeds": [int(value) for value in approximation_seeds],
            "kernel_sampling_policy": (
                "sample non-empty, non-full coalitions without replacement when "
                "M <= 2^p - 2; switch to sampling with replacement for larger M"
            ),
            "kernel_interior_coalition_count": (1 << len(prepared.player_names)) - 2,
            "nmae_denominator": "sum_t sum_i abs(exact_decision_shap[t, i])",
            "exact_abs_shap_sum": float(exact_denominator),
        },
        "arguments": _jsonable_args(args),
        "ems": {
            "coverage_radius_km": float(prepared.config.coverage_radius_km),
            "facility_budget": int(prepared.config.facility_budget),
            "coverage_solver": normalize_ems_coverage_solver(prepared.config.coverage_solver),
            "coverage_matrix_density": float(np.mean(coverage_matrix)),
            "player_count": len(prepared.player_names),
            "player_names": list(prepared.player_names),
            "zip_count": len(prepared.zip_codes),
            "zip_codes": list(prepared.zip_codes),
            "target_columns": list(prepared.target_columns),
        },
        "prepared_data": prepared.preparation_metadata,
        "prediction_metrics": prepared.prediction_metrics,
        "exact_reference": {
            "runtime_seconds": float(exact_reference.runtime_seconds),
            "oracle_calls": int(exact_reference.oracle_calls),
            "coalition_count_per_hour": 1 << len(prepared.player_names),
            "hour_count": len(prepared.explained_timestamps),
            "decision_solve_seconds": float(
                exact_reference.timing["exact_decision_solve_seconds"].sum()
            ),
            "shap_transform_seconds": float(
                exact_reference.timing["exact_shap_transform_seconds"].sum()
            ),
            "seconds_per_oracle": _exact_seconds_per_oracle(exact_reference),
        },
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
        "solver_params": _build_solver_params(prepared.config),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _write_readme(
    path: Path,
    *,
    args: argparse.Namespace,
    raw_metrics: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    exact_denominator: float,
) -> None:
    lines = [
        "# EMS Decision-SHAP Approximation Experiment",
        "",
        "Exact deterministic decision SHAP is computed once and reused as the reference.",
        "",
        f"- Coverage radius / tau: {args.coverage_radius_km:g} km",
        f"- Facility budget: {args.facility_budget}",
        f"- Explained hours: {int(raw_metrics['hour_count'].max())}",
        f"- Sample budgets: {', '.join(str(value) for value in args.sample_budget)}",
        f"- Approximation seeds: {len(raw_metrics.loc[raw_metrics['method'] != 'exact', 'seed'].dropna().unique())}",
        f"- Pooled NMAE denominator: {exact_denominator:.6g}",
        "",
        "## Metric Definitions",
        "",
        "- `nmae`: pooled sum of absolute approximation errors divided by pooled "
        "sum of absolute exact decision-SHAP values.",
        "- `global_top1_match`: whether the top mean-absolute global feature matches "
        "the exact decision-SHAP top feature.",
        "- `global_kendall_tau_b`: Kendall tau-b between exact and approximate global "
        "mean-absolute feature rankings.",
        "- `runtime_seconds`: exact rows are measured standalone exact decision-SHAP "
        "time. Approximation rows are estimated standalone cached runtime: sampled "
        "unique oracle calls priced by the exact run's mean seconds per oracle, plus "
        "the measured approximation transform time.",
        "- `actual_runner_runtime_seconds`: wall time spent by this script after exact "
        "decision games are cached. For approximations this is only the estimator "
        "transform time, so it is not comparable to exact standalone runtime.",
        "- `standalone_cached_oracle_calls`: unique coalition-value calls needed by a "
        "cached standalone approximation implementation. Permutation counts can vary "
        "by seed because sampled path prefixes can duplicate. Kernel counts are "
        "deterministic while `M <= 2^p - 2` because coalitions are sampled without "
        "replacement; for larger `M`, kernel switches to with-replacement sampling and "
        "cached counts can vary by seed.",
        "- `standalone_noncached_oracle_calls`: deterministic draw-budget count without "
        "deduplicating repeated coalitions.",
        "- `actual_runner_oracle_calls`: oracle calls made during the approximation "
        "phase in this runner. It is zero because exact games are intentionally reused.",
        "",
        "## Summary",
        "",
        _markdown_table(
            summary_metrics,
            [
                "method",
                "sample_budget",
                "n_runs",
                "nmae_mean",
                "nmae_ci95_low",
                "nmae_ci95_high",
                "runtime_seconds_mean",
                "actual_runner_runtime_seconds_mean",
                "global_top1_match_mean",
                "global_kendall_tau_b_mean",
                "standalone_cached_oracle_calls_mean",
                "standalone_noncached_oracle_calls_mean",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    selected_columns = [column for column in columns if column in frame.columns]
    header = "| " + " | ".join(selected_columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected_columns) + " |"
    rows = [header, separator]
    for _, row in frame.loc[:, selected_columns].iterrows():
        rows.append(
            "| "
            + " | ".join(_format_markdown_value(row[column]) for column in selected_columns)
            + " |"
        )
    return "\n".join(rows)


def _format_markdown_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
