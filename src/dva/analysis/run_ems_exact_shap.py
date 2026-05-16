from __future__ import annotations

import argparse
from pathlib import Path

from dva.analysis.ems_exact_shap import (
    DEFAULT_BACKGROUND_ROWS,
    DEFAULT_COALITION_BATCH_SIZE,
    DEFAULT_COVERAGE_RADIUS_KM,
    DEFAULT_CVAR_ALPHA,
    DEFAULT_CVAR_SCENARIO_COUNT,
    DEFAULT_DISTANCE_MATRIX_PATH,
    DEFAULT_EXCLUDED_ZIP_CODES,
    DEFAULT_FACILITY_BUDGET,
    DEFAULT_HOLDOUT_HOURS,
    DEFAULT_METADATA_PATH,
    DEFAULT_OBJECTIVE_TOLERANCE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROGRESS_EVERY_COALITIONS,
    DEFAULT_TEST_MONTHS,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_ZONE_ORDER_PATH,
    EmsExactShapConfig,
    SUPPORTED_EMS_COVERAGE_SOLVERS,
    normalize_ems_coverage_solver,
    run_ems_exact_shap,
    write_ems_exact_shap_outputs,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EMS grouped exact-SHAP maximum coverage case study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Wide EMS metadata JSON. Pass a missing path to infer feature columns from X.",
    )
    parser.add_argument("--zone-order-path", type=Path, default=DEFAULT_ZONE_ORDER_PATH)
    parser.add_argument(
        "--distance-matrix-path",
        type=Path,
        default=DEFAULT_DISTANCE_MATRIX_PATH,
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        choices=SUPPORTED_EMS_MODELS,
        default="xgb",
        help="Forecasting model family. Additional choices can be added here later.",
    )
    parser.add_argument(
        "--solver",
        choices=SUPPORTED_EMS_SOLVERS,
        default="gurobi",
        help="EMS facility-location solver for model-driven decisions.",
    )
    parser.add_argument(
        "--coverage-radius-km",
        "--ambulance-distance-km",
        type=float,
        default=DEFAULT_COVERAGE_RADIUS_KM,
        help="Maximum ZIP-centroid distance for a selected ambulance/facility to cover demand.",
    )
    parser.add_argument(
        "--facility-budget",
        "--ambulances",
        type=int,
        default=DEFAULT_FACILITY_BUDGET,
        help="Maximum number of ambulance/facility locations to select.",
    )
    parser.add_argument(
        "--holdout-hours",
        type=int,
        default=DEFAULT_HOLDOUT_HOURS,
        help=(
            "Number of hours to sample without replacement from the holdout/test "
            "month(s) for exact-SHAP explanation."
        ),
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=DEFAULT_TEST_MONTHS,
        help="Number of final calendar month(s) reserved as the holdout/test set.",
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=None,
        help="Optional cap on sampled explanation hours.",
    )
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
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--model-id", default="xgb_001")
    parser.add_argument(
        "--train-sample-rows",
        type=int,
        default=None,
        help="Optional training-row sample for quick experiments.",
    )
    parser.add_argument("--xgb-n-estimators", type=int, default=100)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-verbosity", type=int, default=0)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--mip-gap-abs", type=float, default=1e-9)
    parser.add_argument("--gurobi-threads", type=int, default=1)
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=DEFAULT_CVAR_ALPHA,
        help="CVaR confidence level for residual-bootstrap EMS demand scenarios.",
    )
    parser.add_argument(
        "--cvar-scenario-count",
        type=int,
        default=DEFAULT_CVAR_SCENARIO_COUNT,
        help="Number of residual-bootstrap scenarios per coalition for CVaR SHAP.",
    )
    parser.add_argument(
        "--no-cvar-decision-shap",
        dest="compute_cvar_decision_shap",
        action="store_false",
        default=True,
        help="Skip the CVaR decision-SHAP branch.",
    )
    parser.add_argument(
        "--exclude-zip-codes",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_ZIP_CODES),
        help="ZIP codes expected to be absent from the EMS model universe.",
    )
    parser.add_argument(
        "--objective-tolerance",
        type=float,
        default=DEFAULT_OBJECTIVE_TOLERANCE,
        help="Allowed primary max-coverage objective degradation during deterministic tie-breaking.",
    )
    parser.add_argument(
        "--save-coalition-values",
        action="store_true",
        help="Write coalition_values.csv for diagnostic inspection.",
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
    parser.add_argument(
        "--decision-permutation-shap-samples",
        nargs="+",
        type=int,
        default=[],
        help=(
            "Opt-in sample counts for deterministic decision permutation-SHAP "
            "approximation columns."
        ),
    )
    parser.add_argument(
        "--decision-permutation-shap-seed",
        type=int,
        default=None,
        help=(
            "Random seed for deterministic decision permutation-SHAP sampling. "
            "Defaults to --random-state."
        ),
    )
    parser.add_argument(
        "--decision-kernel-shap-samples",
        nargs="+",
        type=int,
        default=[],
        help=(
            "Opt-in sample counts for deterministic decision kernel-SHAP "
            "approximation columns."
        ),
    )
    parser.add_argument(
        "--decision-kernel-shap-seed",
        type=int,
        default=None,
        help=(
            "Random seed for deterministic decision kernel-SHAP sampling. "
            "Defaults to --random-state."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.model != "xgb":
        raise ValueError(f"Unsupported EMS model: {args.model}")
    coverage_solver = normalize_ems_coverage_solver(args.solver)

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
        n_jobs=args.n_jobs,
        model_id=args.model_id,
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
        solver_seed=args.solver_seed,
        mip_gap=args.mip_gap,
        mip_gap_abs=args.mip_gap_abs,
        gurobi_threads=args.gurobi_threads,
        objective_tolerance=args.objective_tolerance,
        coverage_solver=coverage_solver,
        excluded_zip_codes=tuple(str(zip_code) for zip_code in args.exclude_zip_codes),
        save_coalition_values=args.save_coalition_values,
        compute_ante_decision_shap=args.compute_ante_decision_shap,
        cvar_alpha=args.cvar_alpha,
        cvar_scenario_count=args.cvar_scenario_count,
        compute_cvar_decision_shap=args.compute_cvar_decision_shap,
        decision_permutation_shap_samples=tuple(args.decision_permutation_shap_samples),
        decision_kernel_shap_samples=tuple(args.decision_kernel_shap_samples),
        decision_permutation_shap_seed=args.decision_permutation_shap_seed,
        decision_kernel_shap_seed=args.decision_kernel_shap_seed,
    )
    outputs = run_ems_exact_shap(config)
    write_ems_exact_shap_outputs(outputs, config.outdir)

    holdout_metrics = outputs.prediction_metrics["holdout"]
    print(
        "Holdout prediction metrics "
        f"({args.model}): "
        f"MAE={holdout_metrics['mae']:.6f}, "
        f"MSE={holdout_metrics['mse']:.6f}, "
        f"RMSE={holdout_metrics['rmse']:.6f}",
    )
    print(
        "EMS exact SHAP completed: "
        f"runtime_seconds={outputs.run_metadata['runtime_seconds']:.2f}, "
        f"explained_hours={len(outputs.hourly_shap)}, "
        f"coalitions_per_hour={outputs.run_metadata['coalition_count']}, "
        f"coverage_solver={coverage_solver}, "
        f"outdir={config.outdir}",
    )


if __name__ == "__main__":
    main()
