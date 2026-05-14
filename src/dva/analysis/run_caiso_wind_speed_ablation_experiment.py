from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from dva.analysis.caiso_shap import (
    CaisoShapCaseStudyConfig,
    DEFAULT_INTERACTION_METHOD,
    ParameterPlayerSpec,
    run_caiso_shap_case_study,
    write_caiso_shap_case_study_outputs,
)
from dva.model.storage_dispatch import StorageDispatchParameters


DEFAULT_SOURCE_RESULTS = Path("results/caiso_faith_shap_xgb_lp_no_params")
DEFAULT_OUTDIR = Path("results/caiso_wind_speed_mean_holdout_xgb_lp_no_params")
DEFAULT_ABLATED_FEATURE = "mean_wind_speed"
DEFAULT_VALIDATION_DAYS = 71
BASELINE_SUBDIR = "baseline_recreated"
ABLATION_SUBDIR = "wind_speed_mean_holdout"
HOLDOUT_METRIC_KEYS = (
    "mae",
    "mse",
    "rmse",
    "mean_actual_daily_regret",
    "mean_decision_value_gain",
    "days",
    "targets_per_day",
    "predictions",
)
LOWER_IS_BETTER = {"mae", "mse", "rmse", "mean_actual_daily_regret"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the CAISO XGBoost faith-SHAP baseline, then rerun the same "
            "trained-model experiment after replacing holdout wind speed with "
            "its background-set mean."
        ),
    )
    parser.add_argument(
        "--source-results",
        type=Path,
        default=DEFAULT_SOURCE_RESULTS,
        help="Existing result directory whose run_metadata.json defines the baseline recipe.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Fresh output directory for the comparison experiment.",
    )
    parser.add_argument(
        "--feature",
        default=DEFAULT_ABLATED_FEATURE,
        help="Holdout feature to replace with its background-set mean.",
    )
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for validating recreated baseline holdout metrics.",
    )
    parser.add_argument(
        "--allow-baseline-mismatch",
        action="store_true",
        help="Write outputs even if recreated baseline metrics differ from the source.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help=(
            "Optional smoke-run day limit. Overrides --validation-days and "
            "full source validation is skipped when set."
        ),
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=DEFAULT_VALIDATION_DAYS,
        help=(
            "Number of rows immediately after the training split to explain. "
            "Defaults to the 71-day validation window."
        ),
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help=(
            "Explain the full holdout/test set from the source metadata instead "
            "of the validation window."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip predictive/decision/EAD comparison plot generation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_results = args.source_results
    outdir = args.outdir
    if outdir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {outdir}"
        )

    metadata = _load_json(source_results / "run_metadata.json")
    source_prediction_metrics = _load_json(source_results / "prediction_metrics.json")
    explain_days = _resolve_explain_day_limit(args)
    baseline_config = _build_config_from_metadata(
        metadata,
        outdir=outdir / BASELINE_SUBDIR,
        max_days=explain_days,
    )
    feature_columns = tuple(metadata.get("feature_columns", ()))
    if args.feature not in feature_columns:
        raise ValueError(
            f"Feature {args.feature!r} is not in source feature_columns: "
            + ", ".join(feature_columns)
        )
    ablation_config = dataclasses.replace(
        baseline_config,
        outdir=outdir / ABLATION_SUBDIR,
        holdout_mean_impute_features=(args.feature,),
    )

    window_label = (
        "full holdout/test set"
        if explain_days is None
        else f"first {explain_days} holdout rows (validation window)"
    )
    print(f"Running recreated baseline on {window_label} from source metadata...")
    baseline_outputs = run_caiso_shap_case_study(baseline_config)
    baseline_validation = _validate_recreated_baseline(
        source_prediction_metrics=source_prediction_metrics,
        recreated_prediction_metrics=baseline_outputs.prediction_metrics,
        tolerance=args.baseline_tolerance,
        skip=explain_days is not None,
    )
    if (
        not baseline_validation["skipped"]
        and not baseline_validation["matches_within_tolerance"]
        and not args.allow_baseline_mismatch
    ):
        raise RuntimeError(
            "Recreated baseline does not match source prediction metrics within "
            f"{args.baseline_tolerance}. Pass --allow-baseline-mismatch to write "
            "diagnostic outputs anyway."
        )

    print(
        f"Running {window_label} {args.feature} background-mean ablation..."
    )
    ablation_outputs = run_caiso_shap_case_study(ablation_config)

    summary = _build_experiment_summary(
        source_results=source_results,
        outdir=outdir,
        feature=args.feature,
        explain_days=explain_days,
        baseline_config=baseline_config,
        ablation_outputs=ablation_outputs,
        source_prediction_metrics=source_prediction_metrics,
        baseline_outputs=baseline_outputs,
        baseline_validation=baseline_validation,
    )

    outdir.mkdir(parents=True, exist_ok=False)
    write_caiso_shap_case_study_outputs(
        baseline_outputs,
        outdir / BASELINE_SUBDIR,
    )
    write_caiso_shap_case_study_outputs(
        ablation_outputs,
        outdir / ABLATION_SUBDIR,
    )
    plot_paths: list[str] = []
    if not args.skip_plots:
        from dva.plots.compare_pred_dec import create_comparison_plots

        plot_paths = [
            str(path)
            for run_subdir in (BASELINE_SUBDIR, ABLATION_SUBDIR)
            for path in create_comparison_plots(
                daily_shap_path=outdir / run_subdir / "daily_shap.csv",
                outdir=outdir / "plots" / run_subdir,
            )
        ]
        summary["plot_paths"] = plot_paths
    with (outdir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    baseline_regret = summary["runs"]["baseline_recreated"]["mean_actual_daily_regret"]
    ablated_regret = summary["runs"]["wind_speed_mean_holdout"][
        "mean_actual_daily_regret"
    ]
    regret_delta = summary["delta_wind_speed_mean_holdout_minus_baseline"][
        "mean_actual_daily_regret"
    ]
    print(f"Wrote wind-speed ablation experiment to {outdir}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} comparison plot artifacts under {outdir / 'plots'}")
    print(
        "Mean actual daily regret: "
        f"baseline={baseline_regret:.6f}, "
        f"ablated={ablated_regret:.6f}, "
        f"delta={regret_delta:.6f}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_explain_day_limit(args: argparse.Namespace) -> int | None:
    if args.max_days is not None:
        if args.max_days <= 0:
            raise ValueError("--max-days must be strictly positive when provided.")
        return args.max_days
    if args.full_test:
        return None
    if args.validation_days <= 0:
        raise ValueError("--validation-days must be strictly positive.")
    return args.validation_days


def _build_config_from_metadata(
    metadata: dict[str, Any],
    *,
    outdir: Path,
    max_days: int | None,
) -> CaisoShapCaseStudyConfig:
    if metadata.get("model_name") != "xgb":
        raise ValueError(
            "This experiment is intended for source metadata with model_name='xgb'."
        )

    defaults = CaisoShapCaseStudyConfig()
    xgb_params = metadata.get("xgb_params", {})
    solver_params = metadata.get("solver_params", {})
    source_max_days = metadata.get("max_days")
    return CaisoShapCaseStudyConfig(
        dataset_path=Path(metadata["dataset_path"]),
        holdout_days=int(metadata["holdout_days"]),
        outdir=outdir,
        model_name="xgb",
        random_state=int(metadata.get("random_state", defaults.random_state)),
        n_jobs=int(metadata.get("n_jobs", defaults.n_jobs)),
        mlp_hidden_layer_sizes=tuple(
            int(value)
            for value in metadata.get(
                "mlp_hidden_layer_sizes",
                defaults.mlp_hidden_layer_sizes,
            )
        ),
        mlp_max_iter=int(metadata.get("mlp_max_iter", defaults.mlp_max_iter)),
        xgb_n_estimators=int(
            xgb_params.get("n_estimators", defaults.xgb_n_estimators)
        ),
        xgb_max_depth=int(xgb_params.get("max_depth", defaults.xgb_max_depth)),
        xgb_learning_rate=float(
            xgb_params.get("learning_rate", defaults.xgb_learning_rate)
        ),
        xgb_subsample=float(xgb_params.get("subsample", defaults.xgb_subsample)),
        xgb_colsample_bytree=float(
            xgb_params.get("colsample_bytree", defaults.xgb_colsample_bytree)
        ),
        xgb_reg_lambda=float(
            xgb_params.get("reg_lambda", defaults.xgb_reg_lambda)
        ),
        xgb_verbosity=int(xgb_params.get("verbosity", defaults.xgb_verbosity)),
        learning_rate=metadata.get("learning_rate"),
        mse_learning_rate=metadata.get("mse_learning_rate"),
        spo_learning_rate=metadata.get("spo_learning_rate"),
        training_verbose=bool(
            metadata.get("training_verbose", defaults.training_verbose)
        ),
        training_log_every=metadata.get("training_log_every"),
        spo_processes=metadata.get("spo_processes"),
        spo_warm_start_with_mse=bool(
            metadata.get("spo_warm_start_with_mse", defaults.spo_warm_start_with_mse)
        ),
        solver_seed=int(solver_params.get("Seed", defaults.solver_seed)),
        mip_gap=float(solver_params.get("MIPGap", defaults.mip_gap)),
        mip_gap_abs=float(solver_params.get("MIPGapAbs", defaults.mip_gap_abs)),
        objective_tolerance=float(
            metadata.get("objective_tolerance", defaults.objective_tolerance)
        ),
        max_days=max_days if max_days is not None else source_max_days,
        background_days=int(metadata.get("background_days", defaults.background_days)),
        storage_parameters=_storage_parameters_from_metadata(
            metadata.get("storage_parameters", {}),
            defaults.storage_parameters,
        ),
        interaction_order=metadata.get("interaction_order"),
        interaction_method=metadata.get("interaction_method")
        or DEFAULT_INTERACTION_METHOD,
        parameter_player_spec=_parameter_player_spec_from_metadata(
            metadata.get("parameter_player_spec")
        ),
        compute_ead_decision_shap=True,
    )


def _storage_parameters_from_metadata(
    values: dict[str, Any],
    defaults: StorageDispatchParameters,
) -> StorageDispatchParameters:
    return StorageDispatchParameters(
        energy_capacity=float(values.get("energy_capacity", defaults.energy_capacity)),
        power_limit=float(values.get("power_limit", defaults.power_limit)),
        charge_efficiency=float(
            values.get("charge_efficiency", defaults.charge_efficiency)
        ),
        discharge_efficiency=float(
            values.get("discharge_efficiency", defaults.discharge_efficiency)
        ),
        throughput_penalty=float(
            values.get("throughput_penalty", defaults.throughput_penalty)
        ),
        initial_state_of_charge=float(
            values.get(
                "initial_state_of_charge",
                defaults.initial_state_of_charge,
            )
        ),
        terminal_state_of_charge=float(
            values.get(
                "terminal_state_of_charge",
                defaults.terminal_state_of_charge,
            )
        ),
    )


def _parameter_player_spec_from_metadata(
    values: dict[str, Any] | None,
) -> ParameterPlayerSpec | None:
    if values is None:
        return None
    return ParameterPlayerSpec(**values)


def _validate_recreated_baseline(
    *,
    source_prediction_metrics: dict[str, Any],
    recreated_prediction_metrics: dict[str, Any],
    tolerance: float,
    skip: bool,
) -> dict[str, Any]:
    if skip:
        return {
            "skipped": True,
            "reason": (
                "The recreated run explains a subset of the source holdout rows, "
                "so source full-run metrics are not comparable."
            ),
            "matches_within_tolerance": None,
            "differences": {},
        }

    source_holdout = source_prediction_metrics["holdout"]
    recreated_holdout = recreated_prediction_metrics["holdout"]
    differences: dict[str, Any] = {}
    matches = True
    for key in HOLDOUT_METRIC_KEYS:
        source_value = source_holdout.get(key)
        recreated_value = recreated_holdout.get(key)
        if _is_finite_number(source_value) and _is_finite_number(recreated_value):
            delta = float(recreated_value) - float(source_value)
            within_tolerance = abs(delta) <= tolerance
            matches = matches and within_tolerance
            differences[key] = {
                "source": source_value,
                "recreated": recreated_value,
                "delta": delta,
                "within_tolerance": within_tolerance,
            }
        else:
            same = source_value == recreated_value
            matches = matches and same
            differences[key] = {
                "source": source_value,
                "recreated": recreated_value,
                "matches": same,
            }
    return {
        "skipped": False,
        "tolerance": tolerance,
        "matches_within_tolerance": matches,
        "differences": differences,
    }


def _build_experiment_summary(
    *,
    source_results: Path,
    outdir: Path,
    feature: str,
    explain_days: int | None,
    baseline_config: CaisoShapCaseStudyConfig,
    ablation_outputs: Any,
    source_prediction_metrics: dict[str, Any],
    baseline_outputs: Any,
    baseline_validation: dict[str, Any],
) -> dict[str, Any]:
    source_run = _summarize_prediction_metrics(source_prediction_metrics)
    baseline_run = _summarize_prediction_metrics(baseline_outputs.prediction_metrics)
    ablation_run = _summarize_prediction_metrics(ablation_outputs.prediction_metrics)
    delta = _metric_delta(ablation_run, baseline_run)
    replacement_value = ablation_outputs.run_metadata[
        "holdout_feature_replacements"
    ][feature]
    return {
        "source_results": str(source_results),
        "outdir": str(outdir),
        "experiment": "holdout_background_mean_feature_ablation",
        "evaluation_window": "full_test" if explain_days is None else "validation",
        "explained_rows": ablation_outputs.run_metadata["explain_rows"],
        "requested_explain_days": explain_days,
        "source_holdout_days": baseline_config.holdout_days,
        "ablated_feature": feature,
        "replacement_strategy": "background_mean",
        "replacement_value": replacement_value,
        "background_days": baseline_config.background_days,
        "background_rows": ablation_outputs.run_metadata["background_rows"],
        "background_date_start": ablation_outputs.run_metadata[
            "background_date_start"
        ],
        "background_date_end": ablation_outputs.run_metadata["background_date_end"],
        "baseline_recreation_validation": baseline_validation,
        "runs": {
            "source_existing": source_run,
            "baseline_recreated": baseline_run,
            "wind_speed_mean_holdout": ablation_run,
        },
        "ead_decision_shap": {
            "characteristic_function": "v_ead(S)=J(yhat_N,w(empty))-J(yhat_N,w(S))",
            "baseline_recreated": _summarize_ead_decision_shap(baseline_outputs),
            "wind_speed_mean_holdout": _summarize_ead_decision_shap(ablation_outputs),
        },
        "delta_wind_speed_mean_holdout_minus_baseline": delta,
        "improved_relative_to_oracle": (
            delta.get("mean_actual_daily_regret") is not None
            and delta["mean_actual_daily_regret"] < 0.0
        ),
    }


def _summarize_prediction_metrics(
    prediction_metrics: dict[str, Any],
) -> dict[str, Any]:
    holdout = prediction_metrics["holdout"]
    return {key: holdout.get(key) for key in HOLDOUT_METRIC_KEYS}


def _summarize_ead_decision_shap(outputs: Any) -> dict[str, Any]:
    daily_shap = outputs.daily_shap
    summary_shap = outputs.summary_shap
    if "ead_decision_value_gain" not in daily_shap.columns:
        return {"computed": False}

    top_features = []
    if "ead_decision_mean_abs_shap" in summary_shap.columns:
        top_features = (
            summary_shap.sort_values("ead_decision_mean_abs_shap", ascending=False)
            .loc[:, ["feature", "ead_decision_mean_abs_shap"]]
            .head(5)
            .to_dict(orient="records")
        )
    return {
        "computed": True,
        "mean_ead_decision_value_gain": float(
            daily_shap["ead_decision_value_gain"].mean()
        ),
        "mean_ead_decision_baseline_value": float(
            daily_shap["ead_decision_baseline_value"].mean()
        ),
        "mean_ead_decision_full_value": float(
            daily_shap["ead_decision_full_value"].mean()
        ),
        "top_features_by_mean_abs_ead_shap": top_features,
    }


def _metric_delta(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for key in HOLDOUT_METRIC_KEYS:
        candidate_value = candidate.get(key)
        baseline_value = baseline.get(key)
        if not (
            _is_finite_number(candidate_value)
            and _is_finite_number(baseline_value)
        ):
            deltas[key] = None
            continue
        delta = float(candidate_value) - float(baseline_value)
        deltas[key] = delta
        if key in LOWER_IS_BETTER:
            deltas[f"{key}_improved"] = delta < 0.0
        elif key == "mean_decision_value_gain":
            deltas[f"{key}_improved"] = delta > 0.0
    return deltas


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


if __name__ == "__main__":
    main()
