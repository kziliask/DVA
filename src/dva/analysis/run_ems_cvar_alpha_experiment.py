from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd

from dva.analysis.ems_exact_shap import (
    DEFAULT_BACKGROUND_ROWS,
    DEFAULT_COALITION_BATCH_SIZE,
    DEFAULT_COVERAGE_RADIUS_KM,
    DEFAULT_CVAR_SCENARIO_COUNT,
    DEFAULT_DISTANCE_MATRIX_PATH,
    DEFAULT_EXCLUDED_ZIP_CODES,
    DEFAULT_FACILITY_BUDGET,
    DEFAULT_HOLDOUT_HOURS,
    DEFAULT_METADATA_PATH,
    DEFAULT_OBJECTIVE_TOLERANCE,
    DEFAULT_PROGRESS_EVERY_COALITIONS,
    DEFAULT_TEST_MONTHS,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_ZONE_ORDER_PATH,
    EmsExactShapConfig,
    EmsExactShapOutputs,
    SUPPORTED_EMS_COVERAGE_SOLVERS,
    load_ems_exact_shap_outputs,
    normalize_ems_coverage_solver,
    run_ems_exact_shap,
    write_ems_exact_shap_outputs,
)
from dva.analysis.evaluation_metrics import (
    build_attribution_ranking,
    compute_rank_kendall_tau_from_rankings,
    compute_rank_spearman_from_rankings,
)


DEFAULT_OUTPUT_ROOT = Path("results/ems_cvar_alpha_experiment")
DEFAULT_CVAR_ALPHA_GRID = (0.0, 0.5, 0.8, 0.9, 0.95)
SUPPORTED_EMS_MODELS = ("xgb",)
SUPPORTED_EMS_SOLVERS = (
    *SUPPORTED_EMS_COVERAGE_SOLVERS,
    "gurobi-lp-relaxation",
    "linear-relaxation",
    "lp",
    "lp-relaxation",
    "naive-greedy",
    "greedy-max-cover",
)


@dataclass(frozen=True, slots=True)
class EmsCvarAlphaRunSpec:
    run_label: str
    run_kind: str
    cvar_alpha: float | None
    outdir: Path
    stdout_log_path: Path
    stderr_log_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a normal EMS exact-SHAP experiment and compare it against "
            "CVaR decision-SHAP runs over a small alpha grid."
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
    parser.add_argument(
        "--model",
        choices=SUPPORTED_EMS_MODELS,
        default="xgb",
    )
    parser.add_argument(
        "--solver",
        choices=SUPPORTED_EMS_SOLVERS,
        default="gurobi",
        help="EMS solver for the normal deterministic decision path.",
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=list(DEFAULT_CVAR_ALPHA_GRID),
        help="CVaR alpha values to compare against the normal EMS run.",
    )
    parser.add_argument(
        "--cvar-scenario-count",
        type=int,
        default=DEFAULT_CVAR_SCENARIO_COUNT,
    )
    parser.add_argument(
        "--coverage-radius-km",
        "--ambulance-distance-km",
        type=float,
        default=DEFAULT_COVERAGE_RADIUS_KM,
    )
    parser.add_argument(
        "--facility-budget",
        "--ambulances",
        type=int,
        default=DEFAULT_FACILITY_BUDGET,
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
    parser.add_argument(
        "--progress-every-coalitions",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_COALITIONS,
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--solver-seed", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
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
        "--exclude-zip-codes",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_ZIP_CODES),
    )
    parser.add_argument(
        "--objective-tolerance",
        type=float,
        default=DEFAULT_OBJECTIVE_TOLERANCE,
    )
    parser.add_argument(
        "--save-coalition-values",
        action="store_true",
        help="Write coalition_values.csv for every run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run experiments even when complete outputs already exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.model != "xgb":
        raise ValueError(f"Unsupported EMS model: {args.model}")

    alpha_grid = _resolve_alpha_grid(args.alpha)
    coverage_solver = normalize_ems_coverage_solver(args.solver)
    solver_seed = args.random_state if args.solver_seed is None else args.solver_seed
    out_root = Path(args.out_root)
    run_specs = _build_run_specs(out_root, alpha_grid)

    out_root.mkdir(parents=True, exist_ok=True)
    comparison_dir = out_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    experiment_started_at = time.perf_counter()
    manifest_rows: list[dict[str, Any]] = []
    run_events: list[dict[str, Any]] = []
    outputs_by_label: dict[str, EmsExactShapOutputs] = {}
    reference_explained_hours: tuple[str, ...] | None = None

    try:
        for run_idx, spec in enumerate(run_specs, start=1):
            config = _build_config(args, spec, coverage_solver, solver_seed)
            print(
                f"[{run_idx}/{len(run_specs)}] {spec.run_label}: "
                f"kind={spec.run_kind}, "
                f"alpha={spec.cvar_alpha if spec.cvar_alpha is not None else 'none'}, "
                f"outdir={spec.outdir}",
                flush=True,
            )
            run_started_at = time.perf_counter()
            outputs, loaded_existing = _load_or_run_spec(
                spec,
                config,
                overwrite=bool(args.overwrite),
            )
            run_wall_seconds = time.perf_counter() - run_started_at
            _validate_run_metadata(spec, config, outputs.run_metadata)
            reference_explained_hours = _validate_shared_explained_hours(
                spec,
                outputs.run_metadata,
                reference_explained_hours,
            )
            outputs_by_label[spec.run_label] = outputs
            manifest_row = _build_manifest_row(
                spec,
                config,
                outputs,
                loaded_existing=loaded_existing,
                run_wall_seconds=run_wall_seconds,
            )
            manifest_rows.append(manifest_row)
            run_events.append(
                {
                    "run_label": spec.run_label,
                    "status": "loaded" if loaded_existing else "completed",
                    "wall_seconds": run_wall_seconds,
                    "outdir": str(spec.outdir),
                    "stdout_log_path": str(spec.stdout_log_path),
                    "stderr_log_path": str(spec.stderr_log_path),
                }
            )
            print(
                f"[{run_idx}/{len(run_specs)}] finished {spec.run_label} "
                f"in {run_wall_seconds:.2f}s",
                flush=True,
            )

        normal_outputs = outputs_by_label["normal"]
        cvar_outputs_by_alpha = {
            spec.cvar_alpha: outputs_by_label[spec.run_label]
            for spec in run_specs
            if spec.cvar_alpha is not None
        }
        per_hour = _build_per_hour_comparison(normal_outputs, cvar_outputs_by_alpha)
        alpha_summary = _build_alpha_summary(
            per_hour,
            normal_outputs,
            cvar_outputs_by_alpha,
        )
        run_manifest = pd.DataFrame(manifest_rows)

        run_manifest.to_csv(out_root / "run_manifest.csv", index=False)
        per_hour.to_csv(comparison_dir / "per_hour_comparison.csv", index=False)
        alpha_summary.to_csv(comparison_dir / "alpha_summary.csv", index=False)

        experiment_log = _build_experiment_log(
            args=args,
            alpha_grid=alpha_grid,
            coverage_solver=coverage_solver,
            solver_seed=solver_seed,
            status="completed",
            started_wall_seconds=time.perf_counter() - experiment_started_at,
            manifest_rows=manifest_rows,
            run_events=run_events,
            reference_explained_hours=reference_explained_hours,
            output_paths={
                "run_manifest": out_root / "run_manifest.csv",
                "per_hour_comparison": comparison_dir / "per_hour_comparison.csv",
                "alpha_summary": comparison_dir / "alpha_summary.csv",
            },
        )
        _write_json(out_root / "experiment_log.json", experiment_log)

        print(f"Wrote run manifest to {out_root / 'run_manifest.csv'}")
        print(f"Wrote per-hour comparison to {comparison_dir / 'per_hour_comparison.csv'}")
        print(f"Wrote alpha summary to {comparison_dir / 'alpha_summary.csv'}")
        print(f"Wrote detailed experiment log to {out_root / 'experiment_log.json'}")
    except Exception as exc:
        failure_log = _build_experiment_log(
            args=args,
            alpha_grid=alpha_grid,
            coverage_solver=coverage_solver,
            solver_seed=solver_seed,
            status="failed",
            started_wall_seconds=time.perf_counter() - experiment_started_at,
            manifest_rows=manifest_rows,
            run_events=[
                *run_events,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            ],
            reference_explained_hours=reference_explained_hours,
            output_paths={},
        )
        _write_json(out_root / "experiment_log.json", failure_log)
        raise


def _resolve_alpha_grid(raw_alphas: Sequence[float]) -> tuple[float, ...]:
    if not raw_alphas:
        raise ValueError("At least one CVaR alpha value is required.")
    alphas: list[float] = []
    for raw_alpha in raw_alphas:
        alpha = float(raw_alpha)
        if not 0.0 <= alpha < 1.0:
            raise ValueError("CVaR alpha values must be in [0, 1).")
        if not any(np.isclose(alpha, existing) for existing in alphas):
            alphas.append(alpha)
    return tuple(alphas)


def _build_run_specs(
    out_root: Path,
    alpha_grid: Sequence[float],
) -> list[EmsCvarAlphaRunSpec]:
    logs_dir = out_root / "logs"
    runs_dir = out_root / "runs"
    specs = [
        EmsCvarAlphaRunSpec(
            run_label="normal",
            run_kind="normal",
            cvar_alpha=None,
            outdir=runs_dir / "normal",
            stdout_log_path=logs_dir / "normal_stdout.log",
            stderr_log_path=logs_dir / "normal_stderr.log",
        )
    ]
    for alpha in alpha_grid:
        label = f"cvar_alpha_{_format_alpha_id(alpha)}"
        specs.append(
            EmsCvarAlphaRunSpec(
                run_label=label,
                run_kind="cvar",
                cvar_alpha=float(alpha),
                outdir=runs_dir / label,
                stdout_log_path=logs_dir / f"{label}_stdout.log",
                stderr_log_path=logs_dir / f"{label}_stderr.log",
            )
        )
    return specs


def _build_config(
    args: argparse.Namespace,
    spec: EmsCvarAlphaRunSpec,
    coverage_solver: str,
    solver_seed: int,
) -> EmsExactShapConfig:
    return EmsExactShapConfig(
        x_path=args.x_path,
        y_path=args.y_path,
        metadata_path=args.metadata_path,
        zone_order_path=args.zone_order_path,
        distance_matrix_path=args.distance_matrix_path,
        outdir=spec.outdir,
        holdout_hours=args.holdout_hours,
        test_months=args.test_months,
        max_hours=args.max_hours,
        background_rows=args.background_rows,
        coalition_batch_size=args.coalition_batch_size,
        progress_every_coalitions=args.progress_every_coalitions,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_reg_lambda=args.xgb_reg_lambda,
        xgb_verbosity=args.xgb_verbosity,
        train_sample_rows=args.train_sample_rows,
        coverage_radius_km=args.coverage_radius_km,
        facility_budget=args.facility_budget,
        solver_seed=solver_seed,
        mip_gap=args.mip_gap,
        mip_gap_abs=args.mip_gap_abs,
        gurobi_threads=args.gurobi_threads,
        objective_tolerance=args.objective_tolerance,
        coverage_solver=coverage_solver,
        excluded_zip_codes=tuple(str(zip_code) for zip_code in args.exclude_zip_codes),
        save_coalition_values=args.save_coalition_values,
        cvar_alpha=0.0 if spec.cvar_alpha is None else float(spec.cvar_alpha),
        cvar_scenario_count=args.cvar_scenario_count,
        compute_cvar_decision_shap=spec.run_kind == "cvar",
    )


def _load_or_run_spec(
    spec: EmsCvarAlphaRunSpec,
    config: EmsExactShapConfig,
    *,
    overwrite: bool,
) -> tuple[EmsExactShapOutputs, bool]:
    if not overwrite and _outputs_are_complete(spec.outdir, spec.run_kind == "cvar"):
        outputs = load_ems_exact_shap_outputs(spec.outdir)
        return outputs, True

    spec.stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    spec.outdir.mkdir(parents=True, exist_ok=True)
    with (
        spec.stdout_log_path.open("w", encoding="utf-8") as stdout_handle,
        spec.stderr_log_path.open("w", encoding="utf-8") as stderr_handle,
        contextlib.redirect_stdout(_LogWriter(stdout_handle)),
        contextlib.redirect_stderr(_LogWriter(stderr_handle)),
    ):
        print(
            f"Running {spec.run_label}: kind={spec.run_kind}, "
            f"alpha={spec.cvar_alpha}",
            flush=True,
        )
        outputs = run_ems_exact_shap(config)
        write_ems_exact_shap_outputs(outputs, config.outdir)
    return outputs, False


def _outputs_are_complete(results_dir: Path, expect_cvar: bool) -> bool:
    required_files = [
        "hourly_shap.csv",
        "predictive_zip_shap.csv",
        "coverage_solutions.csv",
        "summary_shap.csv",
        "prediction_metrics.json",
        "evaluation_metrics.json",
        "run_metadata.json",
    ]
    if expect_cvar:
        required_files.append("cvar_summary_shap.csv")
    return all((results_dir / filename).exists() for filename in required_files)


def _validate_run_metadata(
    spec: EmsCvarAlphaRunSpec,
    config: EmsExactShapConfig,
    run_metadata: dict[str, Any],
) -> None:
    expected_values: dict[str, Any] = {
        "compute_cvar_decision_shap": spec.run_kind == "cvar",
        "coverage_solver": config.coverage_solver,
        "coverage_radius_km": float(config.coverage_radius_km),
        "facility_budget": int(config.facility_budget),
        "random_state": int(config.random_state),
        "holdout_hours": int(config.holdout_hours),
        "background_rows": int(config.background_rows),
    }
    if spec.cvar_alpha is not None:
        expected_values["cvar_alpha"] = float(spec.cvar_alpha)
        expected_values["cvar_scenario_count"] = int(config.cvar_scenario_count)

    mismatches = []
    for key, expected in expected_values.items():
        observed = run_metadata.get(key)
        if isinstance(expected, float):
            matches = observed is not None and np.isclose(float(observed), expected)
        else:
            matches = observed == expected
        if not matches:
            mismatches.append(f"{key}: expected {expected!r}, observed {observed!r}")
    if mismatches:
        raise ValueError(
            f"Existing outputs for {spec.run_label} do not match the requested "
            "experiment settings. Re-run with --overwrite. "
            + "; ".join(mismatches)
        )


def _validate_shared_explained_hours(
    spec: EmsCvarAlphaRunSpec,
    run_metadata: dict[str, Any],
    reference_explained_hours: tuple[str, ...] | None,
) -> tuple[str, ...]:
    explained_hours = tuple(str(value) for value in run_metadata["explained_hours"])
    if reference_explained_hours is None:
        return explained_hours
    if explained_hours != reference_explained_hours:
        raise RuntimeError(
            f"{spec.run_label} sampled different explanation hours. Use the same "
            "random_state, holdout_hours, max_hours, and input data across runs."
        )
    return reference_explained_hours


def _build_per_hour_comparison(
    normal_outputs: EmsExactShapOutputs,
    cvar_outputs_by_alpha: dict[float, EmsExactShapOutputs],
) -> pd.DataFrame:
    normal_hourly = normal_outputs.hourly_shap.set_index("timestamp_hour", drop=False)
    player_names = tuple(str(name) for name in normal_outputs.run_metadata["player_names"])
    rows: list[dict[str, Any]] = []
    for alpha, cvar_outputs in sorted(cvar_outputs_by_alpha.items()):
        cvar_hourly = cvar_outputs.hourly_shap.set_index("timestamp_hour", drop=False)
        missing_hours = sorted(set(normal_hourly.index) - set(cvar_hourly.index))
        if missing_hours:
            raise ValueError(
                f"CVaR alpha={alpha:g} is missing explained hours: "
                + ", ".join(str(hour) for hour in missing_hours[:5])
            )

        for timestamp_hour, normal_row in normal_hourly.iterrows():
            cvar_row = cvar_hourly.loc[timestamp_hour]
            normal_shap = np.asarray(
                [normal_row[f"decision_shap_{name}"] for name in player_names],
                dtype=float,
            )
            cvar_shap = np.asarray(
                [cvar_row[f"cvar_decision_shap_{name}"] for name in player_names],
                dtype=float,
            )
            normal_ranking = build_attribution_ranking(normal_shap, player_names)
            cvar_ranking = build_attribution_ranking(cvar_shap, player_names)
            normal_selected = _json_tuple(normal_row["full_selected_zip_codes"])
            cvar_selected = _json_tuple(cvar_row["cvar_full_selected_zip_codes"])
            normal_actual_regret = float(normal_row["actual_regret"])
            cvar_actual_regret = float(cvar_row["cvar_actual_regret"])
            normal_full_value = float(normal_row["decision_full_value"])
            cvar_full_value = float(cvar_row["cvar_decision_full_value"])
            rows.append(
                {
                    "alpha": float(alpha),
                    "timestamp_hour": str(timestamp_hour),
                    "oracle_value": float(normal_row["oracle_value"]),
                    "normal_decision_full_value": normal_full_value,
                    "cvar_decision_full_value": cvar_full_value,
                    "cvar_minus_normal_full_value": cvar_full_value - normal_full_value,
                    "normal_actual_regret": normal_actual_regret,
                    "cvar_actual_regret": cvar_actual_regret,
                    "cvar_minus_normal_actual_regret": (
                        cvar_actual_regret - normal_actual_regret
                    ),
                    "cvar_better_than_normal": cvar_actual_regret < normal_actual_regret,
                    "normal_decision_value_gain": float(
                        normal_row["decision_value_gain"]
                    ),
                    "cvar_decision_value_gain": float(
                        cvar_row["cvar_decision_value_gain"]
                    ),
                    "normal_full_selected_zip_codes": json.dumps(list(normal_selected)),
                    "cvar_full_selected_zip_codes": json.dumps(list(cvar_selected)),
                    "full_selection_changed": normal_selected != cvar_selected,
                    "cvar_full_risk_objective_value": _optional_float(
                        cvar_row.get("cvar_full_risk_objective_value")
                    ),
                    "cvar_baseline_risk_objective_value": _optional_float(
                        cvar_row.get("cvar_baseline_risk_objective_value")
                    ),
                    "decision_vs_cvar_shap_l1": float(
                        np.abs(cvar_shap - normal_shap).sum()
                    ),
                    "decision_vs_cvar_shap_l2": float(
                        np.linalg.norm(cvar_shap - normal_shap)
                    ),
                    "decision_vs_cvar_rank_spearman": (
                        compute_rank_spearman_from_rankings(
                            normal_ranking,
                            cvar_ranking,
                        )
                    ),
                    "decision_vs_cvar_rank_kendall_tau": (
                        compute_rank_kendall_tau_from_rankings(
                            normal_ranking,
                            cvar_ranking,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["alpha", "timestamp_hour"]).reset_index(drop=True)


def _build_alpha_summary(
    per_hour_comparison: pd.DataFrame,
    normal_outputs: EmsExactShapOutputs,
    cvar_outputs_by_alpha: dict[float, EmsExactShapOutputs],
) -> pd.DataFrame:
    normal_prediction_metrics = normal_outputs.prediction_metrics["holdout"]
    rows: list[dict[str, Any]] = []
    for alpha, group in per_hour_comparison.groupby("alpha", sort=True):
        cvar_outputs = cvar_outputs_by_alpha[float(alpha)]
        cvar_metadata = cvar_outputs.run_metadata
        rows.append(
            {
                "alpha": float(alpha),
                "hour_count": int(len(group)),
                "cvar_scenario_count": int(cvar_metadata["cvar_scenario_count"]),
                "normal_run_runtime_seconds": float(
                    normal_outputs.run_metadata["runtime_seconds"]
                ),
                "cvar_run_runtime_seconds": float(cvar_metadata["runtime_seconds"]),
                "holdout_mae": float(normal_prediction_metrics["mae"]),
                "holdout_rmse": float(normal_prediction_metrics["rmse"]),
                "mean_oracle_value": _safe_mean(group["oracle_value"]),
                "mean_normal_decision_full_value": _safe_mean(
                    group["normal_decision_full_value"]
                ),
                "mean_cvar_decision_full_value": _safe_mean(
                    group["cvar_decision_full_value"]
                ),
                "mean_cvar_minus_normal_full_value": _safe_mean(
                    group["cvar_minus_normal_full_value"]
                ),
                "mean_normal_actual_regret": _safe_mean(
                    group["normal_actual_regret"]
                ),
                "mean_cvar_actual_regret": _safe_mean(group["cvar_actual_regret"]),
                "median_cvar_actual_regret": _safe_median(
                    group["cvar_actual_regret"]
                ),
                "mean_cvar_minus_normal_actual_regret": _safe_mean(
                    group["cvar_minus_normal_actual_regret"]
                ),
                "median_cvar_minus_normal_actual_regret": _safe_median(
                    group["cvar_minus_normal_actual_regret"]
                ),
                "cvar_better_hour_share": _safe_mean(
                    group["cvar_better_than_normal"].astype(float)
                ),
                "full_selection_change_share": _safe_mean(
                    group["full_selection_changed"].astype(float)
                ),
                "mean_decision_vs_cvar_shap_l1": _safe_mean(
                    group["decision_vs_cvar_shap_l1"]
                ),
                "mean_decision_vs_cvar_shap_l2": _safe_mean(
                    group["decision_vs_cvar_shap_l2"]
                ),
                "mean_decision_vs_cvar_rank_spearman": _safe_mean(
                    group["decision_vs_cvar_rank_spearman"]
                ),
                "mean_decision_vs_cvar_rank_kendall_tau": _safe_mean(
                    group["decision_vs_cvar_rank_kendall_tau"]
                ),
                "mean_cvar_full_risk_objective_value": _safe_mean(
                    group["cvar_full_risk_objective_value"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_manifest_row(
    spec: EmsCvarAlphaRunSpec,
    config: EmsExactShapConfig,
    outputs: EmsExactShapOutputs,
    *,
    loaded_existing: bool,
    run_wall_seconds: float,
) -> dict[str, Any]:
    holdout_metrics = outputs.prediction_metrics["holdout"]
    return {
        "run_label": spec.run_label,
        "run_kind": spec.run_kind,
        "cvar_alpha": spec.cvar_alpha,
        "compute_cvar_decision_shap": bool(config.compute_cvar_decision_shap),
        "cvar_scenario_count": int(config.cvar_scenario_count),
        "loaded_existing": bool(loaded_existing),
        "run_wall_seconds": float(run_wall_seconds),
        "metadata_runtime_seconds": float(outputs.run_metadata["runtime_seconds"]),
        "outdir": str(spec.outdir),
        "stdout_log_path": str(spec.stdout_log_path),
        "stderr_log_path": str(spec.stderr_log_path),
        "coverage_solver": config.coverage_solver,
        "coverage_radius_km": float(config.coverage_radius_km),
        "facility_budget": int(config.facility_budget),
        "explained_hours": int(outputs.run_metadata["explained_hour_sample_size"]),
        "coalition_count": int(outputs.run_metadata["coalition_count"]),
        "background_rows": int(outputs.run_metadata["background_rows"]),
        "holdout_mae": float(holdout_metrics["mae"]),
        "holdout_rmse": float(holdout_metrics["rmse"]),
    }


def _build_experiment_log(
    *,
    args: argparse.Namespace,
    alpha_grid: Sequence[float],
    coverage_solver: str,
    solver_seed: int,
    status: str,
    started_wall_seconds: float,
    manifest_rows: Sequence[dict[str, Any]],
    run_events: Sequence[dict[str, Any]],
    reference_explained_hours: Sequence[str] | None,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "status": status,
        "runtime_seconds": float(started_wall_seconds),
        "alpha_grid": [float(alpha) for alpha in alpha_grid],
        "coverage_solver": coverage_solver,
        "solver_seed": int(solver_seed),
        "configuration": {
            "x_path": str(args.x_path),
            "y_path": str(args.y_path),
            "metadata_path": str(args.metadata_path),
            "zone_order_path": str(args.zone_order_path),
            "distance_matrix_path": str(args.distance_matrix_path),
            "out_root": str(args.out_root),
            "holdout_hours": int(args.holdout_hours),
            "test_months": int(args.test_months),
            "max_hours": args.max_hours,
            "background_rows": int(args.background_rows),
            "coalition_batch_size": int(args.coalition_batch_size),
            "progress_every_coalitions": int(args.progress_every_coalitions),
            "random_state": int(args.random_state),
            "cvar_scenario_count": int(args.cvar_scenario_count),
            "coverage_radius_km": float(args.coverage_radius_km),
            "facility_budget": int(args.facility_budget),
            "train_sample_rows": args.train_sample_rows,
            "xgb_n_estimators": int(args.xgb_n_estimators),
            "xgb_max_depth": int(args.xgb_max_depth),
            "xgb_learning_rate": float(args.xgb_learning_rate),
            "xgb_subsample": float(args.xgb_subsample),
            "xgb_colsample_bytree": float(args.xgb_colsample_bytree),
            "xgb_reg_lambda": float(args.xgb_reg_lambda),
            "gurobi_threads": int(args.gurobi_threads),
        },
        "shared_explained_hours": list(reference_explained_hours or ()),
        "runs": list(manifest_rows),
        "events": list(run_events),
        "output_paths": {key: str(path) for key, path in output_paths.items()},
    }


def _json_tuple(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return tuple(str(item) for item in parsed)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except TypeError:
        pass
    return float(value)


def _safe_mean(values: Sequence[float] | pd.Series) -> float | None:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return None
    return float(series.mean())


def _safe_median(values: Sequence[float] | pd.Series) -> float | None:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return None
    return float(series.median())


def _format_alpha_id(alpha: float) -> str:
    return f"{float(alpha):.2f}".replace(".", "p")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


class _LogWriter:
    def __init__(self, file_handle: TextIO) -> None:
        self.file_handle = file_handle

    def write(self, value: str) -> int:
        return self.file_handle.write(value)

    def flush(self) -> None:
        self.file_handle.flush()


if __name__ == "__main__":
    main()
