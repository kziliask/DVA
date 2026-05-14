from __future__ import annotations

import argparse
from pathlib import Path

from dva.analysis.caiso_regret_shap import (
    CaisoRegretShapCaseStudyConfig,
    run_caiso_regret_shap_case_study,
    write_caiso_regret_shap_case_study_outputs,
)
from dva.analysis.caiso_shap import DEFAULT_BACKGROUND_DAYS, build_default_storage_parameters
from dva.model.storage_dispatch import StorageDispatchParameters
from dva.model.train import DEFAULT_DATASET_PATH, DEFAULT_HOLDOUT_DAYS, SUPPORTED_MODEL_NAMES


def build_parser() -> argparse.ArgumentParser:
    defaults = build_default_storage_parameters()
    parser = argparse.ArgumentParser(
        description=(
            "Run the CAISO two-stage regret-prediction SHAP experiment. "
            "The first model predicts hourly prices, realized daily regret is "
            "computed from storage dispatch, and a second model with the same "
            "architecture predicts that regret from the original features."
        ),
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--outdir", default="results/caiso_regret_shap_case_study")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODEL_NAMES,
        default="mlp",
        help="First-stage price model. The regret model reuses the same architecture.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--mlp-hidden-units",
        type=int,
        default=256,
        help="Hidden width for mlp, torch_mlp, and spo_mlp architectures.",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=1000,
        help="Maximum training iterations for mlp, torch_mlp, and spo_mlp architectures.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Shared learning rate for torch_mlp and spo_mlp. "
            "For spo_mlp, this applies to both first-stage phases unless "
            "overridden by --mse-lr or --spo-lr."
        ),
    )
    parser.add_argument("--mse-lr", type=float, default=None)
    parser.add_argument("--spo-lr", type=float, default=None)
    parser.add_argument("--training-verbose", action="store_true")
    parser.add_argument("--training-log-every", type=int, default=None)
    parser.add_argument("--spo-processes", type=int, default=None)
    parser.add_argument("--spo-warm-start-with-mse", action="store_true")
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--mip-gap-abs", type=float, default=1e-9)
    parser.add_argument("--objective-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Optional cap on holdout explanation days.",
    )
    parser.add_argument(
        "--background-days",
        type=int,
        default=DEFAULT_BACKGROUND_DAYS,
        help="Calendar days from the end of the training split to use as SHAP background.",
    )
    parser.add_argument(
        "--max-train-days",
        type=int,
        default=None,
        help="Optional cap on rows used to train the second-stage regret model.",
    )
    parser.add_argument(
        "--plots-outdir",
        type=Path,
        default=Path("data/plots/caiso_regret_shap_case_study"),
        help="Directory for the standard CAISO SHAP plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write result tables only; do not generate comparison plots.",
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
    config = CaisoRegretShapCaseStudyConfig(
        dataset_path=args.dataset_path,
        holdout_days=args.holdout_days,
        outdir=args.outdir,
        model_name=args.model,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        mlp_hidden_layer_sizes=(args.mlp_hidden_units,),
        mlp_max_iter=args.mlp_max_iter,
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
        max_train_days=args.max_train_days,
        storage_parameters=storage_parameters,
    )
    outputs = run_caiso_regret_shap_case_study(config)
    write_caiso_regret_shap_case_study_outputs(outputs, config.outdir)

    created_plot_paths = []
    if not args.skip_plots:
        from dva.plots.compare_pred_dec import create_comparison_plots

        created_plot_paths = create_comparison_plots(
            daily_shap_path=Path(config.outdir) / "daily_shap.csv",
            outdir=args.plots_outdir,
        )

    holdout_metrics = outputs.prediction_metrics["holdout"]
    print(
        "Holdout regret prediction metrics "
        f"({outputs.run_metadata['regret_model_name']}): "
        f"MAE={holdout_metrics['mae']:.6f}, "
        f"MSE={holdout_metrics['mse']:.6f}, "
        f"RMSE={holdout_metrics['rmse']:.6f}"
    )
    print(f"Wrote CAISO regret SHAP outputs to {config.outdir}")
    if created_plot_paths:
        print(f"Wrote CAISO regret SHAP plots to {args.plots_outdir}")


if __name__ == "__main__":
    main()
