from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dva.analysis.ems_exact_shap import (
    DEFAULT_COALITION_BATCH_SIZE,
    DEFAULT_CVAR_ALPHA,
    DEFAULT_CVAR_SCENARIO_COUNT,
    DEFAULT_DISTANCE_MATRIX_PATH,
    DEFAULT_EXCLUDED_ZIP_CODES,
    DEFAULT_METADATA_PATH,
    DEFAULT_OBJECTIVE_TOLERANCE,
    DEFAULT_PROGRESS_EVERY_COALITIONS,
    DEFAULT_TEST_MONTHS,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_ZONE_ORDER_PATH,
    EmsExactShapConfig,
    load_ems_exact_shap_outputs,
    normalize_ems_coverage_solver,
    run_ems_exact_shap,
    write_ems_exact_shap_outputs,
)
from dva.analysis.evaluation_metrics import (
    DEFAULT_RBO_DEPTH,
    DEFAULT_RBO_P,
    DEFAULT_TOP_K,
    build_attribution_ranking,
    compute_kendall_tau_correlation,
    compute_normalized_importance_l1,
    compute_rank_kendall_tau_from_rankings,
    compute_rank_spearman_from_rankings,
    compute_spearman_rank_correlation,
    compute_top_k_jaccard,
    compute_truncated_rbo,
    rank_features_from_scores,
)
from dva.case_studies.ems.models import (
    EMS_XGB_MODEL_IDS,
    ems_xgb_config_kwargs,
    make_ems_xgb_model_manifest,
)
from dva.plots.compare_pred_dec import create_comparison_plots


DEFAULT_OUTPUT_ROOT = Path("results/ems_shap_exhaustive_comparison")
DEFAULT_PLOT_ROOT = Path("data/plots/ems_shap_exhaustive_comparison")
DEFAULT_HOLDOUT_HOURS = 100
DEFAULT_BACKGROUND_ROWS = 100
DEFAULT_RANDOM_STATE = 0
DEFAULT_SOLVERS = (
    "naive_greedy",
    "greedy_max_cover",
    "lp_relaxation",
    "exact",
)
DEFAULT_COVERAGE_RADII_KM = (1.0, 2.0, 3.0)
DEFAULT_FACILITY_BUDGETS = (3, 5, 8)
EXPLAINER_FAMILIES = ("predictive", "decision")
_SOLVER_LABELS = {
    "naive_greedy": "naive",
    "greedy_max_cover": "greedy",
    "lp_relaxation": "lp_relaxation",
    "exact": "exact",
}
_SOLVER_ALIASES = {
    "exact": "exact",
    "gurobi": "exact",
    "naive": "naive_greedy",
    "greedy": "greedy_max_cover",
    "gurobi_lp_relaxation": "lp_relaxation",
    "linear_relaxation": "lp_relaxation",
    "lp": "lp_relaxation",
    "lp_relaxation": "lp_relaxation",
    "relaxation": "lp_relaxation",
}


@dataclass(frozen=True, slots=True)
class EmsSweepSetting:
    setting_id: str
    model_id: str
    model_record: Mapping[str, Any]
    coverage_solver: str
    coverage_radius_km: float
    facility_budget: int
    results_dir: Path
    plots_dir: Path


@dataclass(frozen=True, slots=True)
class EmsSweepRunResult:
    setting_id: str
    run_metadata: dict[str, Any]
    plot_count: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the EMS exhaustive SHAP comparison over coverage solvers, "
            "coverage radii, and facility budgets."
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
    parser.add_argument("--plot-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        choices=EMS_XGB_MODEL_IDS,
        help="EMS XGBoost L25 model id to run. Repeat to run a subset.",
    )
    parser.add_argument(
        "--solver",
        action="append",
        default=None,
        help=(
            "Coverage solver to run. Repeat for multiple solvers. "
            "Aliases: naive, greedy, lp, lp-relaxation, exact."
        ),
    )
    parser.add_argument(
        "--coverage-radius-km",
        nargs="+",
        type=float,
        default=list(DEFAULT_COVERAGE_RADII_KM),
        help="Coverage radii to compare.",
    )
    parser.add_argument(
        "--facility-budget",
        nargs="+",
        type=int,
        default=list(DEFAULT_FACILITY_BUDGETS),
        help="Number of chosen EMS facility locations to compare.",
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
        default=0,
        help=(
            "Coalition progress logging interval inside each explained hour. "
            f"Use {DEFAULT_PROGRESS_EVERY_COALITIONS} for the single-run default."
        ),
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--solver-seed", type=int, default=None)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of sweep settings to run concurrently.",
    )
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
    parser.add_argument("--cvar-alpha", type=float, default=DEFAULT_CVAR_ALPHA)
    parser.add_argument(
        "--cvar-scenario-count",
        type=int,
        default=DEFAULT_CVAR_SCENARIO_COUNT,
    )
    parser.add_argument(
        "--no-cvar-decision-shap",
        dest="compute_cvar_decision_shap",
        action="store_false",
        default=True,
        help="Skip the CVaR decision-SHAP branch for each sweep setting.",
    )
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
        "--overwrite",
        action="store_true",
        help="Re-run settings even when complete outputs already exist.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip per-setting and comparison plot generation.",
    )
    parser.add_argument(
        "--no-save-coalition-values",
        dest="save_coalition_values",
        action="store_false",
        default=True,
        help="Skip writing coalition_values.csv for each setting.",
    )
    parser.add_argument(
        "--compute-ante-infodva",
        "--compute-ante-decision-shap",
        dest="compute_ante_decision_shap",
        action="store_true",
        help=(
            "Also compute ante InfoDVA by evaluating each coalition decision "
            "against the full-model demand forecast."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be strictly positive.")
    solver_seed = args.random_state if args.solver_seed is None else args.solver_seed
    solvers = _resolve_solvers(args.solver)
    model_records = _resolve_model_records(args.model_id)
    settings = _build_sweep_settings(
        model_records=model_records,
        solvers=solvers,
        coverage_radii_km=args.coverage_radius_km,
        facility_budgets=args.facility_budget,
        out_root=args.out_root,
        plot_root=args.plot_root,
    )
    _write_manifest(settings, args.out_root / "manifest.csv")

    configs_by_setting = {
        setting.setting_id: _build_setting_config(
            setting,
            args=args,
            solver_seed=solver_seed,
        )
        for setting in settings
    }
    _run_sweep_settings(
        settings,
        configs_by_setting=configs_by_setting,
        overwrite=args.overwrite,
        no_plots=args.no_plots,
        n_jobs=args.n_jobs,
    )

    outputs_by_setting = {}
    reference_explained_hours: tuple[str, ...] | None = None
    for setting in settings:
        outputs = load_ems_exact_shap_outputs(setting.results_dir)
        _validate_existing_setting_metadata(
            setting,
            configs_by_setting[setting.setting_id],
            outputs.run_metadata,
        )
        reference_explained_hours = _validate_shared_explanation_hours(
            setting,
            outputs.run_metadata,
            reference_explained_hours,
        )
        outputs_by_setting[setting.setting_id] = outputs

    comparison_dir = args.out_root / "comparison"
    comparison_plot_dir = args.plot_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_plot_dir.mkdir(parents=True, exist_ok=True)
    setting_summary = _build_setting_metric_summary(settings, outputs_by_setting)
    solver_vs_exact = _build_solver_vs_exact_metrics(settings, outputs_by_setting)
    parameter_pairwise = _build_parameter_pairwise_metrics(settings, outputs_by_setting)
    per_hour_metrics = _build_per_hour_metric_frame(settings, outputs_by_setting)
    trend_metrics = _build_trend_metrics(setting_summary)

    setting_summary.to_csv(comparison_dir / "setting_metric_summary.csv", index=False)
    solver_vs_exact.to_csv(comparison_dir / "solver_vs_exact_metrics.csv", index=False)
    parameter_pairwise.to_csv(
        comparison_dir / "parameter_pairwise_rank_metrics.csv",
        index=False,
    )
    per_hour_metrics.to_csv(comparison_dir / "per_hour_metrics.csv", index=False)
    with (comparison_dir / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "setting_count": len(settings),
                    "model_ids": [str(record["model_id"]) for record in model_records],
                    "solvers": list(solvers),
                    "coverage_radii_km": [float(value) for value in args.coverage_radius_km],
                    "facility_budgets": [int(value) for value in args.facility_budget],
                    "holdout_hours": int(args.holdout_hours),
                    "background_rows": int(args.background_rows),
                    "random_state": int(args.random_state),
                    "solver_seed": int(solver_seed),
                    "n_jobs": int(args.n_jobs),
                    "shared_explained_hours": list(reference_explained_hours or ()),
                    "top_k": DEFAULT_TOP_K,
                    "rbo_depth": DEFAULT_RBO_DEPTH,
                    "rbo_p": DEFAULT_RBO_P,
                },
                "trend_metrics": trend_metrics,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    if not args.no_plots:
        setting_level_summary = setting_summary.drop_duplicates("setting_id").drop(
            columns=["explainer_family"],
            errors="ignore",
        )
        _write_comparison_plots(
            setting_summary=setting_summary,
            setting_level_summary=setting_level_summary,
            per_hour_metrics=per_hour_metrics,
            solver_vs_exact=solver_vs_exact,
            outdir=comparison_plot_dir,
        )

    print(f"Wrote EMS exhaustive comparison manifest to {args.out_root / 'manifest.csv'}")
    print(f"Wrote EMS exhaustive comparison tables to {comparison_dir}")
    if not args.no_plots:
        print(f"Wrote EMS exhaustive comparison plots to {args.plot_root}")


def _build_setting_config(
    setting: EmsSweepSetting,
    *,
    args: argparse.Namespace,
    solver_seed: int,
) -> EmsExactShapConfig:
    return EmsExactShapConfig(
        x_path=args.x_path,
        y_path=args.y_path,
        metadata_path=args.metadata_path,
        zone_order_path=args.zone_order_path,
        distance_matrix_path=args.distance_matrix_path,
        outdir=setting.results_dir,
        holdout_hours=args.holdout_hours,
        test_months=args.test_months,
        max_hours=args.max_hours,
        background_rows=args.background_rows,
        coalition_batch_size=args.coalition_batch_size,
        progress_every_coalitions=args.progress_every_coalitions,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        model_id=setting.model_id,
        **ems_xgb_config_kwargs(setting.model_record),
        xgb_verbosity=args.xgb_verbosity,
        train_sample_rows=args.train_sample_rows,
        coverage_radius_km=setting.coverage_radius_km,
        facility_budget=setting.facility_budget,
        solver_seed=solver_seed,
        mip_gap=args.mip_gap,
        mip_gap_abs=args.mip_gap_abs,
        gurobi_threads=args.gurobi_threads,
        optimization_solver=args.optimization_solver,
        objective_tolerance=args.objective_tolerance,
        coverage_solver=setting.coverage_solver,
        excluded_zip_codes=tuple(str(zip_code) for zip_code in args.exclude_zip_codes),
        save_coalition_values=args.save_coalition_values,
        compute_ante_decision_shap=args.compute_ante_decision_shap,
        cvar_alpha=args.cvar_alpha,
        cvar_scenario_count=args.cvar_scenario_count,
        compute_cvar_decision_shap=args.compute_cvar_decision_shap,
    )


def _run_sweep_settings(
    settings: Sequence[EmsSweepSetting],
    *,
    configs_by_setting: Mapping[str, EmsExactShapConfig],
    overwrite: bool,
    no_plots: bool,
    n_jobs: int,
) -> tuple[EmsSweepRunResult, ...]:
    if n_jobs == 1 or len(settings) <= 1:
        return tuple(
            _run_setting_task(
                index=index,
                total=len(settings),
                setting=setting,
                config=configs_by_setting[setting.setting_id],
                overwrite=overwrite,
                no_plots=no_plots,
            )
            for index, setting in enumerate(settings, start=1)
        )

    worker_count = min(n_jobs, len(settings))
    print(
        f"Running {len(settings)} EMS sweep settings with {worker_count} workers.",
        flush=True,
    )
    results_by_setting: dict[str, EmsSweepRunResult] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_setting_task,
                index=index,
                total=len(settings),
                setting=setting,
                config=configs_by_setting[setting.setting_id],
                overwrite=overwrite,
                no_plots=no_plots,
            ): setting
            for index, setting in enumerate(settings, start=1)
        }
        try:
            for future in as_completed(futures):
                setting = futures[future]
                result = future.result()
                results_by_setting[result.setting_id] = result
                print(
                    f"Completed {setting.setting_id} "
                    f"({len(results_by_setting)}/{len(settings)} settings).",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    return tuple(results_by_setting[setting.setting_id] for setting in settings)


def _run_setting_task(
    *,
    index: int,
    total: int,
    setting: EmsSweepSetting,
    config: EmsExactShapConfig,
    overwrite: bool,
    no_plots: bool,
) -> EmsSweepRunResult:
    print(
        f"[{index}/{total}] {setting.setting_id}: "
        f"model={setting.model_id}, "
        f"solver={setting.coverage_solver}, "
        f"radius={setting.coverage_radius_km:g}km, "
        f"budget={setting.facility_budget}",
        flush=True,
    )
    outputs = _load_or_run_setting(
        setting,
        config,
        overwrite=overwrite,
    )
    plot_count = None
    if not no_plots:
        created_paths = create_comparison_plots(
            daily_shap_path=setting.results_dir / "hourly_shap.csv",
            outdir=setting.plots_dir,
        )
        plot_count = len(created_paths)
        print(
            f"[{index}/{total}] wrote {plot_count} "
            f"per-setting plots to {setting.plots_dir}",
            flush=True,
        )
    return EmsSweepRunResult(
        setting_id=setting.setting_id,
        run_metadata=outputs.run_metadata,
        plot_count=plot_count,
    )


def _resolve_solvers(raw_solvers: Sequence[str] | None) -> tuple[str, ...]:
    if raw_solvers is None:
        return DEFAULT_SOLVERS
    normalized_solvers = []
    for solver_name in raw_solvers:
        solver_key = str(solver_name).strip().lower().replace("-", "_")
        solver_key = _SOLVER_ALIASES.get(solver_key, solver_key)
        normalized_solvers.append(normalize_ems_coverage_solver(solver_key))
    return tuple(dict.fromkeys(normalized_solvers))


def _resolve_model_records(raw_model_ids: Sequence[str] | None) -> tuple[dict[str, Any], ...]:
    manifest = make_ems_xgb_model_manifest()
    if raw_model_ids is None:
        selected_model_ids = list(EMS_XGB_MODEL_IDS)
    else:
        selected_model_ids = list(dict.fromkeys(str(model_id) for model_id in raw_model_ids))
    records_by_id = {
        str(record["model_id"]): dict(record)
        for record in manifest.to_dict(orient="records")
    }
    unknown = sorted(set(selected_model_ids) - set(records_by_id))
    if unknown:
        raise ValueError("Unknown EMS model_id values: " + ", ".join(unknown))
    return tuple(records_by_id[model_id] for model_id in selected_model_ids)


def _build_sweep_settings(
    *,
    model_records: Sequence[Mapping[str, Any]],
    solvers: Sequence[str],
    coverage_radii_km: Sequence[float],
    facility_budgets: Sequence[int],
    out_root: Path,
    plot_root: Path,
) -> list[EmsSweepSetting]:
    settings: list[EmsSweepSetting] = []
    for record in model_records:
        model_id = str(record["model_id"])
        for solver_name in solvers:
            solver_label = _SOLVER_LABELS.get(solver_name, solver_name)
            for radius_km in coverage_radii_km:
                for facility_budget in facility_budgets:
                    setting_id = (
                        f"ems_{model_id}_{solver_label}_"
                        f"radius_{_format_numeric_id(radius_km)}km_"
                        f"budget_{int(facility_budget)}"
                    )
                    settings.append(
                        EmsSweepSetting(
                            setting_id=setting_id,
                            model_id=model_id,
                            model_record=dict(record),
                            coverage_solver=solver_name,
                            coverage_radius_km=float(radius_km),
                            facility_budget=int(facility_budget),
                            results_dir=(
                                out_root / "models" / model_id / "runs" / setting_id
                            ),
                            plots_dir=plot_root / "models" / model_id / setting_id,
                        )
                    )
    return settings


def _format_numeric_id(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _write_manifest(settings: Sequence[EmsSweepSetting], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "setting_id": setting.setting_id,
            "model_id": setting.model_id,
            "model_name": setting.model_record.get("model_name"),
            "xgb_n_estimators": setting.model_record.get("n_estimators"),
            "xgb_max_depth": setting.model_record.get("max_depth"),
            "xgb_learning_rate": setting.model_record.get("learning_rate"),
            "xgb_subsample": setting.model_record.get("subsample"),
            "xgb_colsample_bytree": setting.model_record.get("colsample_bytree"),
            "xgb_reg_lambda": setting.model_record.get("reg_lambda"),
            "results_dir": str(setting.results_dir),
            "plots_dir": str(setting.plots_dir),
            "coverage_solver": setting.coverage_solver,
            "coverage_solver_label": _SOLVER_LABELS.get(
                setting.coverage_solver,
                setting.coverage_solver,
            ),
            "coverage_radius_km": setting.coverage_radius_km,
            "facility_budget": setting.facility_budget,
        }
        for setting in settings
    ]
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def _load_or_run_setting(
    setting: EmsSweepSetting,
    config: EmsExactShapConfig,
    *,
    overwrite: bool,
):
    if not overwrite and _setting_outputs_are_complete(setting.results_dir, config):
        print(f"Loading existing outputs from {setting.results_dir}", flush=True)
        outputs = load_ems_exact_shap_outputs(setting.results_dir)
        _validate_existing_setting_metadata(setting, config, outputs.run_metadata)
        return outputs

    outputs = run_ems_exact_shap(config)
    write_ems_exact_shap_outputs(outputs, config.outdir)
    return outputs


def _setting_outputs_are_complete(
    results_dir: Path,
    config: EmsExactShapConfig,
) -> bool:
    required_files = [
        "hourly_shap.csv",
        "predictive_zip_shap.csv",
        "coverage_solutions.csv",
        "summary_shap.csv",
        "prediction_metrics.json",
        "evaluation_metrics.json",
        "run_metadata.json",
    ]
    if config.compute_cvar_decision_shap:
        required_files.append("cvar_summary_shap.csv")
    return all((results_dir / filename).exists() for filename in required_files)


def _validate_existing_setting_metadata(
    setting: EmsSweepSetting,
    config: EmsExactShapConfig,
    run_metadata: dict[str, Any],
) -> None:
    expected_values = {
        "model_id": setting.model_id,
        "coverage_solver": setting.coverage_solver,
        "coverage_radius_km": float(config.coverage_radius_km),
        "facility_budget": int(config.facility_budget),
        "random_state": int(config.random_state),
        "background_rows": int(config.background_rows),
        "holdout_hours": int(config.holdout_hours),
        "compute_cvar_decision_shap": bool(config.compute_cvar_decision_shap),
    }
    if config.compute_cvar_decision_shap:
        expected_values["cvar_alpha"] = float(config.cvar_alpha)
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
            f"Existing outputs at {setting.results_dir} do not match the requested "
            "EMS comparison settings. Re-run with --overwrite. "
            + "; ".join(mismatches)
        )


def _validate_shared_explanation_hours(
    setting: EmsSweepSetting,
    run_metadata: dict[str, Any],
    reference_explained_hours: tuple[str, ...] | None,
) -> tuple[str, ...]:
    explained_hours = tuple(str(value) for value in run_metadata["explained_hours"])
    if reference_explained_hours is None:
        return explained_hours
    if explained_hours != reference_explained_hours:
        raise RuntimeError(
            f"{setting.setting_id} sampled different holdout hours from the first "
            "setting. Use the same random_state, holdout_hours, and max_hours."
        )
    return reference_explained_hours


def _build_setting_metric_summary(
    settings: Sequence[EmsSweepSetting],
    outputs_by_setting: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for setting in settings:
        outputs = outputs_by_setting[setting.setting_id]
        hourly = outputs.hourly_shap
        feature_names = tuple(outputs.run_metadata["player_names"])
        prediction_metrics = outputs.prediction_metrics["holdout"]
        rank_agreement = _build_metric_summary_from_series(hourly["abs_rank_kendall_tau"])
        feature_corrs = _featurewise_method_correlations(hourly, feature_names)
        global_rank_agreement = _global_predictive_decision_rank_agreement(
            hourly,
            feature_names,
        )
        common = {
            "setting_id": setting.setting_id,
            "model_id": setting.model_id,
            "coverage_solver": setting.coverage_solver,
            "coverage_solver_label": _SOLVER_LABELS.get(
                setting.coverage_solver,
                setting.coverage_solver,
            ),
            "coverage_radius_km": setting.coverage_radius_km,
            "facility_budget": setting.facility_budget,
            "holdout_hours": int(outputs.run_metadata["explained_hour_sample_size"]),
            "background_rows": int(outputs.run_metadata["background_rows"]),
            "random_state": int(outputs.run_metadata["random_state"]),
            "holdout_mae": prediction_metrics["mae"],
            "holdout_mse": prediction_metrics["mse"],
            "holdout_rmse": prediction_metrics["rmse"],
            "mean_actual_regret": _safe_mean(hourly["actual_regret"]),
            "median_actual_regret": _safe_median(hourly["actual_regret"]),
            "mean_oracle_value": _safe_mean(hourly["oracle_value"]),
            "mean_decision_full_value": _safe_mean(hourly["decision_full_value"]),
            "mean_decision_value_gain": _safe_mean(hourly["decision_value_gain"]),
            "mean_abs_rank_kendall_tau": rank_agreement["mean"],
            "median_abs_rank_kendall_tau": rank_agreement["median"],
            "global_abs_rank_kendall_tau": global_rank_agreement,
            "mean_featurewise_predictive_decision_pearson": _safe_mean(
                pd.Series(feature_corrs)
            ),
            "runtime_seconds": outputs.run_metadata["runtime_seconds"],
            "coverage_matrix_density": outputs.run_metadata["coverage_matrix_density"],
            "results_dir": str(setting.results_dir),
            "plots_dir": str(setting.plots_dir),
        }
        for explainer_family in EXPLAINER_FAMILIES:
            rows.append(
                {
                    **common,
                    "explainer_family": explainer_family,
                    "decision_insertion_auc_mean": _metric_mean(
                        outputs.evaluation_metrics,
                        f"{explainer_family}_decision_insertion_auc",
                    ),
                    "decision_insertion_auc_median": _metric_median(
                        outputs.evaluation_metrics,
                        f"{explainer_family}_decision_insertion_auc",
                    ),
                    "decision_insertion_auc_coverage": _metric_coverage(
                        outputs.evaluation_metrics,
                        f"{explainer_family}_decision_insertion_auc",
                    ),
                    "decision_deletion_auc_mean": _metric_mean(
                        outputs.evaluation_metrics,
                        f"{explainer_family}_decision_deletion_auc",
                    ),
                    "decision_deletion_auc_median": _metric_median(
                        outputs.evaluation_metrics,
                        f"{explainer_family}_decision_deletion_auc",
                    ),
                    "decision_deletion_auc_coverage": _metric_coverage(
                        outputs.evaluation_metrics,
                        f"{explainer_family}_decision_deletion_auc",
                    ),
                    "decision_infidelity_mean": _metric_mean(
                        outputs.evaluation_metrics,
                        f"{explainer_family}_decision_infidelity",
                    ),
                    "decision_infidelity_median": _metric_median(
                        outputs.evaluation_metrics,
                        f"{explainer_family}_decision_infidelity",
                    ),
                    "decision_infidelity_coverage": _metric_coverage(
                        outputs.evaluation_metrics,
                        f"{explainer_family}_decision_infidelity",
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        [
            "model_id",
            "coverage_solver_label",
            "coverage_radius_km",
            "facility_budget",
            "explainer_family",
        ]
    ).reset_index(drop=True)


def _build_solver_vs_exact_metrics(
    settings: Sequence[EmsSweepSetting],
    outputs_by_setting: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    settings_by_key = {
        (
            setting.model_id,
            setting.coverage_solver,
            setting.coverage_radius_km,
            setting.facility_budget,
        ): setting
        for setting in settings
    }
    exact_solver = "exact"
    for setting in settings:
        if setting.coverage_solver == exact_solver:
            continue
        exact_setting = settings_by_key.get(
            (
                setting.model_id,
                exact_solver,
                setting.coverage_radius_km,
                setting.facility_budget,
            )
        )
        if exact_setting is None:
            continue
        rows.extend(
            _build_setting_pair_rows(
                left_setting=setting,
                right_setting=exact_setting,
                outputs_by_setting=outputs_by_setting,
                comparison_kind="solver_vs_exact",
            )
        )
    return pd.DataFrame(rows)


def _build_parameter_pairwise_metrics(
    settings: Sequence[EmsSweepSetting],
    outputs_by_setting: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(
        [
            {
                "setting_id": setting.setting_id,
                "model_id": setting.model_id,
                "coverage_solver": setting.coverage_solver,
                "coverage_radius_km": setting.coverage_radius_km,
                "facility_budget": setting.facility_budget,
            }
            for setting in settings
        ]
    )
    setting_by_id = {setting.setting_id: setting for setting in settings}
    for group_key, group in frame.groupby(
        ["model_id", "coverage_solver", "facility_budget"]
    ):
        model_id, solver, budget = cast(tuple[Any, Any, Any], group_key)
        sorted_group = group.sort_values("coverage_radius_km")
        for _, left_row, right_row in _neighbor_rows(sorted_group):
            rows.extend(
                _build_setting_pair_rows(
                    left_setting=setting_by_id[str(left_row["setting_id"])],
                    right_setting=setting_by_id[str(right_row["setting_id"])],
                    outputs_by_setting=outputs_by_setting,
                    comparison_kind="radius_step",
                    sweep_id=f"{model_id}_{solver}_budget_{int(budget)}_radius",
                )
            )
    for group_key, group in frame.groupby(
        ["model_id", "coverage_solver", "coverage_radius_km"]
    ):
        model_id, solver, radius = cast(tuple[Any, Any, Any], group_key)
        sorted_group = group.sort_values("facility_budget")
        for _, left_row, right_row in _neighbor_rows(sorted_group):
            rows.extend(
                _build_setting_pair_rows(
                    left_setting=setting_by_id[str(left_row["setting_id"])],
                    right_setting=setting_by_id[str(right_row["setting_id"])],
                    outputs_by_setting=outputs_by_setting,
                    comparison_kind="budget_step",
                    sweep_id=(
                        f"{model_id}_{solver}_"
                        f"radius_{_format_numeric_id(float(radius))}_budget"
                    ),
                )
            )
    return pd.DataFrame(rows)


def _neighbor_rows(group: pd.DataFrame):
    records = list(group.iterrows())
    for left, right in zip(records, records[1:]):
        yield left[0], left[1], right[1]


def _build_setting_pair_rows(
    *,
    left_setting: EmsSweepSetting,
    right_setting: EmsSweepSetting,
    outputs_by_setting: dict[str, Any],
    comparison_kind: str,
    sweep_id: str | None = None,
) -> list[dict[str, Any]]:
    left_outputs = outputs_by_setting[left_setting.setting_id]
    right_outputs = outputs_by_setting[right_setting.setting_id]
    left_hours = tuple(
        str(value) for value in left_outputs.run_metadata["explained_hours"]
    )
    right_hours = tuple(
        str(value) for value in right_outputs.run_metadata["explained_hours"]
    )
    if left_hours != right_hours:
        raise ValueError("Setting pair must share explained hours for comparison.")
    feature_names = tuple(left_outputs.run_metadata["player_names"])
    if feature_names != tuple(right_outputs.run_metadata["player_names"]):
        raise ValueError("Setting pair must share player names for comparison.")

    rows = []
    solution_match_rates = _solution_match_rates(
        left_outputs.coverage_solutions,
        right_outputs.coverage_solutions,
    )
    for explainer_family in EXPLAINER_FAMILIES:
        left_global = _global_importance(
            left_outputs.hourly_shap,
            feature_names,
            explainer_family,
        )
        right_global = _global_importance(
            right_outputs.hourly_shap,
            feature_names,
            explainer_family,
        )
        left_ranking = rank_features_from_scores(left_global, feature_names)
        right_ranking = rank_features_from_scores(right_global, feature_names)
        hourly_rank_tau = _hourly_pairwise_rank_agreement(
            left_outputs.hourly_shap,
            right_outputs.hourly_shap,
            feature_names,
            explainer_family,
        )
        rows.append(
            {
                "comparison_kind": comparison_kind,
                "sweep_id": sweep_id,
                "model_id": left_setting.model_id,
                "left_setting_id": left_setting.setting_id,
                "right_setting_id": right_setting.setting_id,
                "left_solver": left_setting.coverage_solver,
                "right_solver": right_setting.coverage_solver,
                "left_radius_km": left_setting.coverage_radius_km,
                "right_radius_km": right_setting.coverage_radius_km,
                "left_facility_budget": left_setting.facility_budget,
                "right_facility_budget": right_setting.facility_budget,
                "explainer_family": explainer_family,
                "top_k_jaccard_5": compute_top_k_jaccard(
                    left_ranking,
                    right_ranking,
                    k=DEFAULT_TOP_K,
                ),
                "rbo_10": compute_truncated_rbo(
                    left_ranking,
                    right_ranking,
                    depth=DEFAULT_RBO_DEPTH,
                    p=DEFAULT_RBO_P,
                ),
                "rank_spearman": compute_rank_spearman_from_rankings(
                    left_ranking,
                    right_ranking,
                ),
                "rank_kendall_tau": compute_rank_kendall_tau_from_rankings(
                    left_ranking,
                    right_ranking,
                ),
                "normalized_importance_l1": compute_normalized_importance_l1(
                    left_global,
                    right_global,
                    feature_names,
                ),
                "hourly_rank_kendall_tau_mean": _safe_mean(hourly_rank_tau),
                "hourly_rank_kendall_tau_median": _safe_median(hourly_rank_tau),
                "full_selected_match_rate": solution_match_rates[
                    "full_selected_match_rate"
                ],
                "full_covered_match_rate": solution_match_rates[
                    "full_covered_match_rate"
                ],
                "left_decision_infidelity_mean": _metric_mean(
                    left_outputs.evaluation_metrics,
                    f"{explainer_family}_decision_infidelity",
                ),
                "right_decision_infidelity_mean": _metric_mean(
                    right_outputs.evaluation_metrics,
                    f"{explainer_family}_decision_infidelity",
                ),
                "left_decision_insertion_auc_mean": _metric_mean(
                    left_outputs.evaluation_metrics,
                    f"{explainer_family}_decision_insertion_auc",
                ),
                "right_decision_insertion_auc_mean": _metric_mean(
                    right_outputs.evaluation_metrics,
                    f"{explainer_family}_decision_insertion_auc",
                ),
                "left_decision_deletion_auc_mean": _metric_mean(
                    left_outputs.evaluation_metrics,
                    f"{explainer_family}_decision_deletion_auc",
                ),
                "right_decision_deletion_auc_mean": _metric_mean(
                    right_outputs.evaluation_metrics,
                    f"{explainer_family}_decision_deletion_auc",
                ),
                "left_actual_regret_mean": _safe_mean(
                    left_outputs.hourly_shap["actual_regret"]
                ),
                "right_actual_regret_mean": _safe_mean(
                    right_outputs.hourly_shap["actual_regret"]
                ),
            }
        )
    return rows


def _build_per_hour_metric_frame(
    settings: Sequence[EmsSweepSetting],
    outputs_by_setting: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    metric_columns = (
        "timestamp_hour",
        "coverage_radius_km",
        "facility_budget",
        "decision_full_value",
        "oracle_value",
        "actual_regret",
        "predictive_decision_deletion_auc",
        "predictive_decision_insertion_auc",
        "predictive_decision_infidelity",
        "decision_decision_deletion_auc",
        "decision_decision_insertion_auc",
        "decision_decision_infidelity",
        "abs_rank_spearman",
        "abs_rank_kendall_tau",
    )
    for setting in settings:
        hourly = outputs_by_setting[setting.setting_id].hourly_shap
        available_columns = [column for column in metric_columns if column in hourly.columns]
        frame = hourly.loc[:, available_columns].copy()
        frame["setting_id"] = setting.setting_id
        frame["model_id"] = setting.model_id
        frame["coverage_solver"] = setting.coverage_solver
        frame["coverage_solver_label"] = _SOLVER_LABELS.get(
            setting.coverage_solver,
            setting.coverage_solver,
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_trend_metrics(setting_summary: pd.DataFrame) -> dict[str, Any]:
    trend_metrics: dict[str, Any] = {}
    if setting_summary.empty:
        return trend_metrics
    for group_key, group in setting_summary.groupby(
        ["model_id", "coverage_solver", "facility_budget", "explainer_family"],
        sort=True,
    ):
        model_id, solver, budget, explainer = cast(tuple[Any, Any, Any, Any], group_key)
        key = f"{model_id}|{solver}|budget={int(budget)}|{explainer}|radius"
        sorted_group = group.sort_values("coverage_radius_km")
        trend_metrics[key] = _metric_trend_payloads(
            sorted_group,
            parameter_column="coverage_radius_km",
        )
    for group_key, group in setting_summary.groupby(
        ["model_id", "coverage_solver", "coverage_radius_km", "explainer_family"],
        sort=True,
    ):
        model_id, solver, radius, explainer = cast(tuple[Any, Any, Any, Any], group_key)
        key = f"{model_id}|{solver}|radius={float(radius):g}|{explainer}|budget"
        sorted_group = group.sort_values("facility_budget")
        trend_metrics[key] = _metric_trend_payloads(
            sorted_group,
            parameter_column="facility_budget",
        )
    return trend_metrics


def _metric_trend_payloads(
    frame: pd.DataFrame,
    *,
    parameter_column: str,
) -> dict[str, Any]:
    return {
        "decision_insertion_auc": _trend_payload(
            frame[parameter_column],
            frame["decision_insertion_auc_mean"],
        ),
        "decision_deletion_auc": _trend_payload(
            frame[parameter_column],
            frame["decision_deletion_auc_mean"],
        ),
        "decision_infidelity": _trend_payload(
            frame[parameter_column],
            frame["decision_infidelity_mean"],
        ),
    }


def _trend_payload(
    parameter_values: Sequence[float] | pd.Series,
    metric_values: Sequence[float] | pd.Series,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "parameter_value": pd.to_numeric(parameter_values, errors="coerce"),
            "metric_value": pd.to_numeric(metric_values, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 3:
        return {"spearman": None, "kendall_tau": None, "setting_count": int(len(frame))}
    return {
        "spearman": compute_spearman_rank_correlation(
            frame["parameter_value"],
            frame["metric_value"],
        ),
        "kendall_tau": compute_kendall_tau_correlation(
            frame["parameter_value"],
            frame["metric_value"],
        ),
        "setting_count": int(len(frame)),
    }


def _write_comparison_plots(
    *,
    setting_summary: pd.DataFrame,
    setting_level_summary: pd.DataFrame,
    per_hour_metrics: pd.DataFrame,
    solver_vs_exact: pd.DataFrame,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    _plot_metric_grid(
        setting_summary,
        outdir / "decision_infidelity_by_setting.png",
        y_column="decision_infidelity_mean",
        ylabel="Mean decision infidelity",
        title="Decision infidelity across EMS solver and coverage settings",
    )
    _plot_metric_grid(
        setting_summary,
        outdir / "decision_deletion_auc_by_setting.png",
        y_column="decision_deletion_auc_mean",
        ylabel="Mean deletion AUC",
        title="Decision deletion AUC across EMS solver and coverage settings",
    )
    _plot_metric_grid(
        setting_level_summary,
        outdir / "kendall_tau_by_setting.png",
        y_column="mean_abs_rank_kendall_tau",
        ylabel="Mean Kendall tau-b",
        title="Predictive-vs-decision rank agreement across EMS settings",
    )
    _plot_metric_grid(
        setting_level_summary,
        outdir / "actual_regret_by_setting.png",
        y_column="mean_actual_regret",
        ylabel="Mean realized regret",
        title="Realized EMS regret across solver and coverage settings",
    )
    _plot_metric_grid(
        setting_level_summary,
        outdir / "runtime_seconds_by_setting.png",
        y_column="runtime_seconds",
        ylabel="Runtime seconds",
        title="Runtime across EMS solver and coverage settings",
    )
    _plot_hourly_decision_vs_predictive_kendall_scatters(per_hour_metrics, outdir)
    if not solver_vs_exact.empty:
        _plot_solver_vs_exact(
            solver_vs_exact,
            outdir / "solver_vs_exact_rank_kendall_tau.png",
        )


def _plot_metric_grid(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    y_column: str,
    ylabel: str,
    title: str,
) -> None:
    plot_frame = frame.dropna(subset=[y_column]).copy()
    if plot_frame.empty:
        return
    family_values = (
        list(plot_frame["explainer_family"].dropna().unique())
        if "explainer_family" in plot_frame.columns
        else [None]
    )
    if not family_values:
        family_values = [None]
    fig, axes = plt.subplots(
        1,
        len(family_values),
        figsize=(6.5 * len(family_values), 5.2),
        squeeze=False,
        sharey=True,
    )
    for ax, family in zip(axes[0], family_values, strict=True):
        family_frame = (
            plot_frame[plot_frame["explainer_family"].eq(family)]
            if family is not None
            else plot_frame
        )
        for group_key, group in family_frame.groupby(
            ["coverage_solver_label", "facility_budget"],
            sort=True,
        ):
            solver_label, budget = cast(tuple[Any, Any], group_key)
            sorted_group = group.sort_values("coverage_radius_km")
            ax.plot(
                sorted_group["coverage_radius_km"],
                sorted_group[y_column],
                marker="o",
                label=f"{solver_label}, k={int(budget)}",
            )
        ax.set_xlabel("Coverage radius (km)")
        ax.set_ylabel(ylabel)
        if family is not None:
            ax.set_title(str(family))
        ax.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_solver_vs_exact(frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = frame.dropna(subset=["rank_kendall_tau"]).copy()
    if plot_frame.empty:
        return
    plot_frame["label"] = plot_frame.apply(
        lambda row: (
            f"{_SOLVER_LABELS.get(str(row['left_solver']), row['left_solver'])} "
            f"r={float(row['left_radius_km']):g}, k={int(row['left_facility_budget'])}, "
            f"{row['explainer_family']}"
        ),
        axis=1,
    )
    plot_frame = plot_frame.sort_values("rank_kendall_tau")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(plot_frame))))
    ax.barh(plot_frame["label"], plot_frame["rank_kendall_tau"])
    ax.set_xlabel("Global rank Kendall tau-b vs exact solver")
    ax.set_ylabel("Heuristic setting")
    ax.set_title("Heuristic SHAP ranking agreement with exact EMS solver")
    ax.axvline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_hourly_decision_vs_predictive_kendall_scatters(
    frame: pd.DataFrame,
    outdir: Path,
) -> None:
    plot_frame = frame.dropna(subset=["abs_rank_kendall_tau"]).copy()
    if plot_frame.empty:
        return
    solver_order = [
        solver_label
        for solver_label in ("naive", "greedy", "lp_relaxation", "exact")
        if solver_label in set(plot_frame["coverage_solver_label"])
    ]

    _plot_decision_vs_predictive_kendall_scatter(
        plot_frame,
        outdir / "decision_vs_predictive_rank_kendall_tau_by_solver.png",
        group_column="coverage_solver_label",
        group_order=solver_order,
        group_labels={
            "naive": "Naive",
            "greedy": "Greedy",
            "lp_relaxation": "LP relaxation",
            "exact": "Exact",
        },
        xlabel="Coverage solver",
        title="Hourly decision SHAP agreement with predictive SHAP by solver",
    )
    _plot_decision_vs_predictive_kendall_scatter(
        plot_frame,
        outdir / "decision_vs_predictive_rank_kendall_tau_by_radius.png",
        group_column="coverage_radius_km",
        group_order=sorted(plot_frame["coverage_radius_km"].dropna().unique()),
        group_labels=lambda value: f"{float(value):g} km",
        xlabel="Coverage radius",
        title="Hourly decision SHAP agreement with predictive SHAP by radius",
    )
    _plot_decision_vs_predictive_kendall_scatter(
        plot_frame,
        outdir / "decision_vs_predictive_rank_kendall_tau_by_facility_budget.png",
        group_column="facility_budget",
        group_order=sorted(plot_frame["facility_budget"].dropna().unique()),
        group_labels=lambda value: f"{int(value)}",
        xlabel="Number of locations",
        title="Hourly decision SHAP agreement with predictive SHAP by number of locations",
    )


def _plot_decision_vs_predictive_kendall_scatter(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    group_column: str,
    group_order: Sequence[Any],
    group_labels: dict[Any, str] | Callable[[Any], str],
    xlabel: str,
    title: str,
) -> None:
    rng = np.random.default_rng(23)
    point_color = "#264653"
    mean_color = "#e76f51"
    x_positions = np.arange(len(group_order))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for x_position, group_value in zip(x_positions, group_order, strict=True):
        values = pd.to_numeric(
            frame.loc[
                frame[group_column].eq(group_value),
                "abs_rank_kendall_tau",
            ],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        jitter = rng.normal(0.0, 0.075, size=len(values))
        ax.scatter(
            np.full(len(values), x_position) + jitter,
            values.to_numpy(dtype=float),
            s=12,
            alpha=0.18,
            color=point_color,
            edgecolor="none",
            zorder=2,
        )
        ax.scatter(
            [x_position],
            [float(values.mean())],
            marker="D",
            s=110,
            color=mean_color,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )

    labels: list[str] = [
        str(group_labels[value] if isinstance(group_labels, dict) else group_labels(value))
        for value in group_order
    ]
    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x_positions, labels=labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Hourly Kendall tau vs predictive SHAP")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=point_color,
            label="Decision SHAP hour",
            markersize=6,
            alpha=0.35,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            color=mean_color,
            markeredgecolor="black",
            label="Group mean",
            markersize=8,
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _metric_mean(evaluation_metrics: dict[str, Any], metric_name: str) -> float | None:
    metric = evaluation_metrics.get(metric_name, {})
    value = metric.get("mean") if isinstance(metric, dict) else None
    return None if value is None else float(value)


def _metric_median(evaluation_metrics: dict[str, Any], metric_name: str) -> float | None:
    metric = evaluation_metrics.get(metric_name, {})
    value = metric.get("median") if isinstance(metric, dict) else None
    return None if value is None else float(value)


def _metric_coverage(evaluation_metrics: dict[str, Any], metric_name: str) -> float | None:
    metric = evaluation_metrics.get(metric_name, {})
    value = metric.get("coverage") if isinstance(metric, dict) else None
    return None if value is None else float(value)


def _safe_mean(values: Sequence[float] | pd.Series) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def _safe_median(values: Sequence[float] | pd.Series) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.median())


def _build_metric_summary_from_series(values: pd.Series) -> dict[str, float | None]:
    return {
        "mean": _safe_mean(values),
        "median": _safe_median(values),
    }


def _global_importance(
    hourly_shap: pd.DataFrame,
    feature_names: Sequence[str],
    explainer_family: str,
) -> dict[str, float]:
    return {
        feature_name: float(
            hourly_shap[f"{explainer_family}_shap_{feature_name}"].abs().mean()
        )
        for feature_name in feature_names
    }


def _global_predictive_decision_rank_agreement(
    hourly_shap: pd.DataFrame,
    feature_names: Sequence[str],
) -> float | None:
    predictive = _global_importance(hourly_shap, feature_names, "predictive")
    decision = _global_importance(hourly_shap, feature_names, "decision")
    predictive_ranking = rank_features_from_scores(predictive, feature_names)
    decision_ranking = rank_features_from_scores(decision, feature_names)
    return compute_rank_kendall_tau_from_rankings(predictive_ranking, decision_ranking)


def _featurewise_method_correlations(
    hourly_shap: pd.DataFrame,
    feature_names: Sequence[str],
) -> list[float]:
    correlations = []
    for feature_name in feature_names:
        predictive = pd.to_numeric(
            hourly_shap[f"predictive_shap_{feature_name}"],
            errors="coerce",
        )
        decision = pd.to_numeric(
            hourly_shap[f"decision_shap_{feature_name}"],
            errors="coerce",
        )
        finite = ~(predictive.isna() | decision.isna())
        if finite.sum() < 2:
            continue
        if predictive[finite].nunique() <= 1 or decision[finite].nunique() <= 1:
            continue
        correlations.append(float(predictive[finite].corr(decision[finite])))
    return correlations


def _hourly_pairwise_rank_agreement(
    left_hourly: pd.DataFrame,
    right_hourly: pd.DataFrame,
    feature_names: Sequence[str],
    explainer_family: str,
) -> pd.Series:
    merged = left_hourly.merge(
        right_hourly,
        on="timestamp_hour",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    values = []
    for _, row in merged.iterrows():
        left_values = [
            float(row[f"{explainer_family}_shap_{feature_name}_left"])
            for feature_name in feature_names
        ]
        right_values = [
            float(row[f"{explainer_family}_shap_{feature_name}_right"])
            for feature_name in feature_names
        ]
        left_ranking = build_attribution_ranking(left_values, feature_names)
        right_ranking = build_attribution_ranking(right_values, feature_names)
        values.append(compute_rank_kendall_tau_from_rankings(left_ranking, right_ranking))
    return pd.Series(values, dtype=float)


def _solution_match_rates(
    left_solutions: pd.DataFrame,
    right_solutions: pd.DataFrame,
) -> dict[str, float | None]:
    left_full = _full_solution_frame(left_solutions)
    right_full = _full_solution_frame(right_solutions)
    if left_full.empty or right_full.empty:
        return {
            "full_selected_match_rate": None,
            "full_covered_match_rate": None,
        }
    merged = left_full.merge(
        right_full,
        on="timestamp_hour",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if merged.empty:
        return {
            "full_selected_match_rate": None,
            "full_covered_match_rate": None,
        }
    return {
        "full_selected_match_rate": float(
            (
                merged["selected_facility_zip_codes_left"]
                == merged["selected_facility_zip_codes_right"]
            ).mean()
        ),
        "full_covered_match_rate": float(
            (merged["covered_zip_codes_left"] == merged["covered_zip_codes_right"]).mean()
        ),
    }


def _full_solution_frame(solutions: pd.DataFrame) -> pd.DataFrame:
    frame = solutions[solutions["solution_type"].eq("full_model")].copy()
    return frame.loc[
        :,
        ["timestamp_hour", "selected_facility_zip_codes", "covered_zip_codes"],
    ]


if __name__ == "__main__":
    main()
