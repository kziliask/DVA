from __future__ import annotations

import argparse

from dva.analysis.caiso_shap import (
    CaisoShapCaseStudyConfig,
    DEFAULT_BACKGROUND_DAYS,
    DEFAULT_CAISO_MODEL_NAME,
    DEFAULT_INTERACTION_METHOD,
    ParameterPlayerSpec,
    SUPPORTED_INTERACTION_METHODS,
    build_default_storage_parameters,
    run_caiso_shap_case_study,
    write_caiso_shap_case_study_outputs,
)
from dva.model.train import DEFAULT_DATASET_PATH, DEFAULT_HOLDOUT_DAYS, SUPPORTED_MODEL_NAMES
from dva.model.storage_dispatch import StorageDispatchParameters


def build_parser() -> argparse.ArgumentParser:
    defaults = build_default_storage_parameters()
    parser = argparse.ArgumentParser(
        description="Run the CAISO predictive SHAP and decision-value SHAP case study.",
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--outdir", default="results/caiso_shap_case_study")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODEL_NAMES,
        default=DEFAULT_CAISO_MODEL_NAME,
        help="Predictive model to train before running the SHAP case study.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--mlp-hidden-units",
        type=int,
        default=256,
        help="Hidden width for the shallow MLP used when --model mlp, --model torch_mlp, or --model spo_mlp.",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=1000,
        help="Maximum training iterations for the shallow MLP used when --model mlp, --model torch_mlp, or --model spo_mlp.",
    )
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-weight-decay", type=float, default=0.0)
    parser.add_argument("--mlp-batch-size", type=int, default=None)
    parser.add_argument("--mlp-early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--mlp-activation",
        choices=("relu", "gelu"),
        default="relu",
    )
    parser.add_argument("--mlp-batch-norm", action="store_true")
    parser.add_argument("--xgb-n-estimators", type=int, default=100)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-verbosity", type=int, default=0)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Shared learning rate for torch_mlp and spo_mlp. "
            "For spo_mlp, this applies to both phases unless overridden by "
            "--mse-lr or --spo-lr."
        ),
    )
    parser.add_argument(
        "--mse-lr",
        type=float,
        default=None,
        help="Optional MSE-phase learning rate override for --model spo_mlp.",
    )
    parser.add_argument(
        "--spo-lr",
        type=float,
        default=None,
        help="Optional SPO+-phase learning rate override for --model spo_mlp.",
    )
    parser.add_argument(
        "--training-verbose",
        action="store_true",
        help="Print progress logs during torch_mlp or spo_mlp training.",
    )
    parser.add_argument(
        "--training-log-every",
        type=int,
        default=None,
        help="Log every N training epochs when --training-verbose is enabled.",
    )
    parser.add_argument(
        "--spo-processes",
        type=int,
        default=None,
        help=(
            "Number of PyEPO worker processes for SPO+ training. "
            "Defaults to a small multicore setting when --model spo_mlp."
        ),
    )
    parser.add_argument(
        "--spo-warm-start-with-mse",
        action="store_true",
        help=(
            "For --model spo_mlp, run an MSE pretraining phase for "
            "--mlp-max-iter epochs before the SPO+ phase."
        ),
    )
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--mip-gap-abs", type=float, default=1e-9)
    parser.add_argument("--objective-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--background-days",
        type=int,
        default=DEFAULT_BACKGROUND_DAYS,
        help="Calendar days from the end of the training split to use as SHAP background.",
    )
    parser.add_argument(
        "--holdout-mean-impute-feature",
        action="append",
        default=[],
        help=(
            "Feature to replace in the holdout/explanation set with its mean over "
            "the selected background rows. Repeat to impute multiple features."
        ),
    )
    parser.add_argument(
        "--compute-ead-shap",
        action="store_true",
        help=(
            "Also compute ex ante decision SHAP using the full model prediction "
            "inside the decision characteristic function."
        ),
    )
    parser.add_argument(
        "--interaction-order",
        type=int,
        default=None,
        help="Optional interaction order. Leave unset to skip interaction outputs.",
    )
    parser.add_argument(
        "--interaction-method",
        choices=SUPPORTED_INTERACTION_METHODS,
        default=DEFAULT_INTERACTION_METHOD,
        help="Interaction index method to use when --interaction-order is set.",
    )
    parser.add_argument(
        "--param-players",
        choices=("none", "all"),
        default="none",
        help="Whether to include storage parameters as additional coalition players.",
    )
    parser.add_argument("--energy-capacity", type=float, default=defaults.energy_capacity)
    parser.add_argument("--power-limit", type=float, default=defaults.power_limit)
    parser.add_argument("--charge-efficiency", type=float, default=defaults.charge_efficiency)
    parser.add_argument(
        "--discharge-efficiency",
        type=float,
        default=defaults.discharge_efficiency,
    )
    parser.add_argument(
        "--throughput-penalty",
        type=float,
        default=defaults.throughput_penalty,
    )
    parser.add_argument(
        "--initial-soc",
        type=float,
        default=defaults.initial_state_of_charge,
    )
    parser.add_argument(
        "--terminal-soc",
        type=float,
        default=defaults.terminal_state_of_charge,
    )
    parser.add_argument(
        "--throughput-penalty-baseline",
        type=float,
        default=0.0,
        help="Baseline throughput penalty when that parameter player is OUT.",
    )
    parser.add_argument(
        "--charge-efficiency-baseline",
        type=float,
        default=1.0,
        help="Baseline charge efficiency when the joint efficiency player is OUT.",
    )
    parser.add_argument(
        "--discharge-efficiency-baseline",
        type=float,
        default=1.0,
        help="Baseline discharge efficiency when the joint efficiency player is OUT.",
    )
    parser.add_argument(
        "--energy-capacity-baseline",
        type=float,
        default=1e6,
        help="Baseline energy capacity when that parameter player is OUT.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    storage_parameters = StorageDispatchParameters(
        energy_capacity=args.energy_capacity,
        power_limit=args.power_limit,
        charge_efficiency=args.charge_efficiency,
        discharge_efficiency=args.discharge_efficiency,
        throughput_penalty=args.throughput_penalty,
        initial_state_of_charge=args.initial_soc,
        terminal_state_of_charge=args.terminal_soc,
    )
    parameter_player_spec = None
    if args.param_players == "all":
        parameter_player_spec = ParameterPlayerSpec(
            throughput_penalty_is_player=True,
            throughput_penalty_baseline=args.throughput_penalty_baseline,
            efficiency_is_player=True,
            charge_efficiency_baseline=args.charge_efficiency_baseline,
            discharge_efficiency_baseline=args.discharge_efficiency_baseline,
            energy_capacity_is_player=True,
            energy_capacity_baseline=args.energy_capacity_baseline,
        )
    config = CaisoShapCaseStudyConfig(
        dataset_path=args.dataset_path,
        holdout_days=args.holdout_days,
        outdir=args.outdir,
        model_name=args.model,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        mlp_hidden_layer_sizes=(args.mlp_hidden_units,),
        mlp_max_iter=args.mlp_max_iter,
        mlp_dropout=args.mlp_dropout,
        mlp_weight_decay=args.mlp_weight_decay,
        mlp_batch_size=args.mlp_batch_size,
        mlp_early_stopping_patience=args.mlp_early_stopping_patience,
        mlp_activation=args.mlp_activation,
        mlp_batch_norm=args.mlp_batch_norm,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_reg_lambda=args.xgb_reg_lambda,
        xgb_verbosity=args.xgb_verbosity,
        learning_rate=args.lr,
        mse_learning_rate=args.mse_lr,
        spo_learning_rate=args.spo_lr,
        training_verbose=args.training_verbose,
        training_log_every=args.training_log_every,
        spo_processes=args.spo_processes,
        spo_warm_start_with_mse=args.spo_warm_start_with_mse,
        solver_seed=args.solver_seed,
        mip_gap=args.mip_gap,
        mip_gap_abs=args.mip_gap_abs,
        objective_tolerance=args.objective_tolerance,
        max_days=args.max_days,
        background_days=args.background_days,
        holdout_mean_impute_features=tuple(args.holdout_mean_impute_feature),
        compute_ead_decision_shap=args.compute_ead_shap,
        storage_parameters=storage_parameters,
        interaction_order=args.interaction_order,
        interaction_method=args.interaction_method,
        parameter_player_spec=parameter_player_spec,
    )
    outputs = run_caiso_shap_case_study(config)
    write_caiso_shap_case_study_outputs(outputs, config.outdir)
    holdout_metrics = outputs.prediction_metrics["holdout"]
    print(
        "Holdout prediction metrics "
        f"({config.model_name}): "
        f"MAE={holdout_metrics['mae']:.6f}, "
        f"MSE={holdout_metrics['mse']:.6f}, "
        f"RMSE={holdout_metrics['rmse']:.6f}"
    )
    print(f"Wrote CAISO SHAP case study outputs to {config.outdir}")


if __name__ == "__main__":
    main()
